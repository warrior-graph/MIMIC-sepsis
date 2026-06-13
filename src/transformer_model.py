import torch
import torch.nn as nn
import math
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler
from typing import Dict, Tuple, Optional, List, Union

# ── GPU-aware defaults ────────────────────────────────────────────────────────
_CUDA = torch.cuda.is_available()
_PIN_MEMORY = _CUDA
_NUM_WORKERS = 4 if _CUDA else 0


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class TimeSeriesTransformer(nn.Module):
    def __init__(self,
                 input_dim: Optional[int] = None,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 nhead: int = 8,
                 dropout: float = 0.2,
                 task_type: str = 'classification'):
        super().__init__()
        self.task_type = task_type
        self.device = torch.device('cuda' if _CUDA else 'cpu')
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        
        # Store input_dim for later initialization
        self.input_dim = input_dim
        self.initialized = False
        
        # Initialize model if input_dim is provided
        if input_dim is not None:
            self._initialize_model(input_dim)
            self.initialized = True
    
    def _initialize_model(self, input_dim):
        """Initialize the model architecture with the given dimensions"""
        # Input projection
        self.input_proj = nn.Linear(input_dim, self.hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.hidden_dim)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.nhead,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, self.num_layers)
        
        # Output layers
        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1)
        )
        
        if self.task_type == 'classification':
            self.output_activation = nn.Sigmoid()
        else:
            self.output_activation = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the transformer model
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Output tensor of shape (batch_size,)
        """
        # Project input to hidden dimension
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x)
        
        # Take the output from the last timestep
        x = x[:, -1, :]
        
        # Pass through output layers
        x = self.output_layer(x)
        
        # Apply activation function
        x = self.output_activation(x)
        
        return x.squeeze()
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            batch_size: int = 32,
            epochs: int = 30,
            learning_rate: float = 0.001,
            weight_decay: float = 1e-4,
            patience: int = 5,
            grad_clip: float = 1.0,
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        """Train the Transformer model with AMP, early stopping, weight decay, and LR scheduling.

        Args:
            X: Input features of shape (n_samples, n_timesteps, n_features)
            y: Target values of shape (n_samples,)
            batch_size: Batch size for training
            epochs: Maximum training epochs
            learning_rate: Initial learning rate for Adam
            weight_decay: L2 regularization coefficient
            patience: Early stopping patience (epochs without val loss improvement)
            grad_clip: Max gradient norm (0 disables clipping)
            validation_data: Optional (X_val, y_val) — enables early stopping and LR scheduling
        """
        # Initialize model if not already done
        if not self.initialized:
            self.input_dim = X.shape[2]
            self._initialize_model(self.input_dim)
            self.initialized = True

        self.to(self.device)

        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=_PIN_MEMORY,
            num_workers=_NUM_WORKERS,
            persistent_workers=(_NUM_WORKERS > 0),
        )

        val_dataloader = None
        if validation_data is not None:
            X_val, y_val = validation_data
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(y_val)
            )
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                pin_memory=_PIN_MEMORY,
                num_workers=_NUM_WORKERS,
                persistent_workers=(_NUM_WORKERS > 0),
            )

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.BCELoss() if self.task_type == 'classification' else nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=3, factor=0.5
        )

        use_amp = _CUDA
        scaler = GradScaler('cuda', enabled=use_amp)

        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        self.train()
        for epoch in range(epochs):
            total_loss = 0.0

            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)

                optimizer.zero_grad()

                with autocast('cuda', enabled=use_amp):
                    outputs = self(batch_X)
                    loss = criterion(outputs, batch_y)

                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()

            train_loss = total_loss / len(dataloader)

            if val_dataloader is not None:
                val_loss = self._validate(val_dataloader, criterion)
                scheduler.step(val_loss)
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in self.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f'Early stopping at epoch {epoch+1} (patience={patience})')
                        self.load_state_dict(best_state)
                        break
            elif (epoch + 1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.4f}')
    
    def _validate(self, val_dataloader, criterion):
        """Run validation and return validation loss"""
        self.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_dataloader:
                batch_X = batch_X.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                
                with autocast('cuda', enabled=_CUDA):
                    outputs = self(batch_X)
                    loss = criterion(outputs, batch_y)
                total_val_loss += loss.item()
        
        self.train()
        return total_val_loss / len(val_dataloader)
    
    def predict(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Generate predictions using the trained model
        
        Args:
            X: Input features of shape (n_samples, n_timesteps, n_features)
            batch_size: Batch size for prediction
            
        Returns:
            Predictions array of shape (n_samples,)
        """
        self.to(self.device)
        self.eval()
        
        # Convert input to PyTorch tensor
        X_tensor = torch.FloatTensor(X)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=_PIN_MEMORY,
            num_workers=_NUM_WORKERS,
            persistent_workers=(_NUM_WORKERS > 0),
        )
        
        predictions = []
        with torch.no_grad():
            for batch_X, in dataloader:
                batch_X = batch_X.to(self.device, non_blocking=True)

                # Use AMP autocast for inference too
                with autocast('cuda', enabled=_CUDA):
                    outputs = self(batch_X)
                
                # Ensure outputs are properly shaped before converting to numpy
                if outputs.ndim == 0:
                    outputs = outputs.unsqueeze(0)
                    
                # Convert to numpy and append to predictions
                batch_preds = outputs.float().cpu().numpy()
                
                # Ensure batch_preds is always at least 1D
                if batch_preds.ndim == 0:
                    batch_preds = np.array([batch_preds.item()])
                    
                predictions.append(batch_preds)
        
        # Concatenate all batch predictions
        if predictions:
            return np.concatenate(predictions)
        else:
            return np.array([])
