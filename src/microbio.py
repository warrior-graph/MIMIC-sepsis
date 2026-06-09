"""
MIMIC-IV Sepsis Cohort Extraction.

This file is sourced and modified from: https://github.com/matthieukomorowski/AI_Clinician
"""

import argparse
import os

import pandas as pd
import psycopg2 as pg


parser = argparse.ArgumentParser()
parser.add_argument("-u", "--username",  help="Username used to access the MIMIC Database", type=str)
parser.add_argument("-p", "--password",  help="User's password for MIMIC Database", type=str)
pargs = parser.parse_args()

# Initializing database connection
conn = pg.connect("dbname='mimiciv' user={0} host='localhost' options='--search_path=mimimciv' password={1}".format(pargs.username,pargs.password))

# Path for processed data storage
exportdir = os.path.join(os.getcwd(),'processed_files')

if not os.path.exists(exportdir):
    os.makedirs(exportdir)

# 2. microbio (Microbiologyevents)
# extract(epoch from charttime) : The number of seconds since 1970-01-01 00:00:00 UTC
query = """
select subject_id, hadm_id, extract(epoch from charttime) as charttime, extract(epoch from chartdate) as chartdate 
from mimiciv_hosp.microbiologyevents
"""

# Wrap the query in a COPY TO STDOUT command, formatting it as a pipe-separated CSV
copy_sql = f"COPY ({query}) TO STDOUT WITH CSV HEADER DELIMITER '|';"

output_file = os.path.join(exportdir, 'microbio.csv')

# Open the file and stream the data directly into it
with open(output_file, 'w') as f:
    cursor = conn.cursor()
    cursor.copy_expert(copy_sql, f)
    cursor.close()

conn.close()