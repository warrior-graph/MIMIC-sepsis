import argparse
import os

import pandas as pd
import psycopg2 as pg


parser = argparse.ArgumentParser()
parser.add_argument("-u", "--username", help="Username used to access the MIMIC Database", type=str)
parser.add_argument("-p", "--password", help="User's password for MIMIC Database", type=str)
pargs = parser.parse_args()

# Initializing database connection
conn = pg.connect("dbname='mimiciv' user={0} host='localhost' options='--search_path=mimimciv' password={1}".format(pargs.username,pargs.password))

# Path for processed data storage
exportdir = os.path.join(os.getcwd(),'processed_files')

if not os.path.exists(exportdir):
    os.makedirs(exportdir)

# Extraction of sub-tables
# There are 43 tables in the Mimic III database. 
# 26 unique tables; the other 17 are partitions of chartevents that are not to be queried directly 
# See: https://mit-lcp.github.io/mimic-schema-spy/
# We create 15 sub-tables when extracting from the database

# From each table we extract subject ID, admission ID, ICU stay ID 
# and relevant times to assist in joining these tables
# All other specific information extracted will be documented before each section of the following code.


# NOTE: The next three tables are built to help identify when a patient may be 
# considered to be septic, using the Sepsis 3 criteria

# 0. icustay mappings
query = """
select * 
from mimiciv_icu.icustays
order by subject_id, hadm_id, stay_id
"""

# Wrap the query in a COPY TO STDOUT command, formatting it as a pipe-separated CSV
copy_sql = f"COPY ({query}) TO STDOUT WITH CSV HEADER DELIMITER '|';"

output_file = os.path.join(exportdir, 'icustays.csv')

# Open the file and stream the data directly into it
with open(output_file, 'w') as f:
    cursor = conn.cursor()
    cursor.copy_expert(copy_sql, f)
    cursor.close()

conn.close()