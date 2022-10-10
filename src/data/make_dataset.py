# Navigate file system
import os
import glob

####### Helper functions to combine files #######

# Combine csv files from a directory to one file
def make_csv_from_csvs(dir):
    files = glob.glob(os.path.join(dir, '*.csv'))
    return files

# Combine json files from a directory to one file
# Convert is to a csv file
def make_csv_from_jsons(dir):
    pass