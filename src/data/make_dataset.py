import os
import glob

#def test():
#    print('Make a dataset')

# Combine csv files from a directory to one file
def make_csv_from_csvs(dir):
    files = glob.glob(os.path.join(dir, '*.csv'))
    return files

# Combine json files from a directory to one file
# Convert is to a csv file
def make_csv_from_jsons(dir):
    pass