import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler
import numpy as np
from typing import Dict, Optional, Tuple

# ── GPU-aware defaults ────────────────────────────────────────────────────────
_CUDA = torch.cuda.is_available()
_PIN_MEMORY = _CUDA
_NUM_WORKERS = 4 if _CUDA else 0


class LSTMModel(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 task_type: str = 'classification',
                 output_dim: int = 1):
        """
        Parameters
        ----------
        input_dim : int
            Number of input features per timestep.
        hidden_dim : int
            LSTM hidden state size.
        num_layers : int
            Number of stacked LSTM layers.
        dropout : float
        task_type : str
            'classification' or 'regression'
        output_dim : int
            1 for single-step output (default, backward-compatible).
            H > 1 for multi-step output — forward() returns shape (N, H).
        """
        super().__init__()
        self.task_type = task_type
        self.output_dim = output_dim
        self.device = torch.device('cuda' if _CUDA else 'cpu')

        # LSTM for processing input sequence
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        # Output layers — last Linear produces output_dim values
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

        if task_type == 'classification':
            self.output_activation = nn.Sigmoid()
        else:
            self.output_activation = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, input_dim)
        lstm_out, _ = self.lstm(x)
        # Take the last hidden state
        last_hidden = lstm_out[:, -1, :]

        output = self.output_layer(last_hidden)          # (N, output_dim)
        output = self.output_activation(output)
        # For single-step (output_dim=1) squeeze to (N,) for backward compatibility
        if self.output_dim == 1:
            return output.squeeze(-1)
        return output                                    # (N, H) for multi-step

    def _validate(self, val_dataloader: DataLoader, criterion: nn.Module) -> float:
        """Run one pass over val_dataloader and return mean loss."""
        self.eval()
        total = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_dataloader:
                batch_X = batch_X.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                with autocast('cuda', enabled=_CUDA):
                    outputs = self(batch_X)
                    loss = criterion(outputs, batch_y)
                total += loss.item()
        self.train()
        return total / len(val_dataloader)

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            batch_size: int = 32,
            epochs: int = 30,
            learning_rate: float = 0.001,
            weight_decay: float = 1e-4,
            patience: int = 5,
            grad_clip: float = 1.0,
            val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            random_state: int = None):
        """Train the LSTM model with AMP, early stopping, weight decay, and LR scheduling.

        Parameters
        ----------
        X : np.ndarray, shape (N, T, F)
        y : np.ndarray, shape (N,) for single-step or (N, H) for multi-step
        val_data : (X_val, y_val) arrays for early stopping / LR scheduling.
            When provided, training stops early if val loss doesn't improve
            for ``patience`` consecutive epochs and restores the best weights.
        grad_clip : float
            Max gradient norm for clipping (0 disables).
        """
        self.to(self.device)

        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)
            if _CUDA:
                torch.cuda.manual_seed(random_state)
                torch.cuda.manual_seed_all(random_state)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

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
        if val_data is not None:
            X_val, y_val = val_data
            val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
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

    def predict(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """Generate predictions using the trained model"""
        self.to(self.device)
        self.eval()

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

                with autocast('cuda', enabled=_CUDA):
                    outputs = self(batch_X)

                if outputs.ndim == 0:
                    outputs = outputs.unsqueeze(0)

                batch_preds = outputs.float().cpu().numpy()

                if batch_preds.ndim == 0:
                    batch_preds = np.array([batch_preds.item()])

                predictions.append(batch_preds)

        if predictions:
            return np.concatenate(predictions)
        else:
            return np.array([])
