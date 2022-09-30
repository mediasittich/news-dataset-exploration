import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'src/data'))
help('modules')

import glob
import pandas as pd

from clean_text import clean_txt, calculate_word_count

DATA_DIR = os.path.join(os.getcwd(), 'data')
RAW_DATA_DIR = os.path.join(os.getcwd(), 'data/raw')

NYT_DATA_DIR = os.path.join(RAW_DATA_DIR, 'nyt_headlines')
GUARDIAN_DATA_DIR = os.path.join(RAW_DATA_DIR, 'guardian')
TWITTER_GU_DATA_DIR = os.path.join(RAW_DATA_DIR, 'gu_twitter')

if not os.path.exists(os.path.join(DATA_DIR, 'interim')):
    os.mkdir(os.path.join(DATA_DIR, 'interim'))
INTERIM_DIR = os.path.join(DATA_DIR, 'interim')

if not os.path.exists(os.path.join(DATA_DIR, 'processed')):
    os.mkdir(os.path.join(DATA_DIR, 'processed'))
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

#############################################################
#                   NYT Dataset
#############################################################
# Load NYT data
nyt_files = glob.glob(os.path.join(NYT_DATA_DIR, '*.csv'))

for file in nyt_files:
    print(f"Processing file {file} ...")
    nyt_df = pd.read_csv(file)

    # select only doc_type == article & material_type == News
    nyt_df = nyt_df[(nyt_df['doc_type'] == 'article') & (nyt_df['material_type'] == 'News')]
    # reset index on dataframe
    nyt_df.reset_index(drop=True, inplace=True)
    # drop columns: 'doc_type', 'material_type', 'snippet', 'lead_paragraph'
    nyt_df.drop(['doc_type', 'material_type', 'snippet', 'lead_paragraph'], axis=1, inplace=True)

    # clean up headline and abstract
    nyt_df['headline'] = nyt_df['headline'].fillna('')
    nyt_df['abstract'] = nyt_df['abstract'].fillna('')
    nyt_df['cl_headline'] = nyt_df.apply(lambda x: clean_txt(x['headline']), axis=1)
    nyt_df['cl_abstract'] = nyt_df.apply(lambda x: clean_txt(x['abstract']), axis=1)
    nyt_df.drop(['headline', 'abstract'], axis=1, inplace=True)

    # calculate word counts for both columns
    nyt_df['word_count_headline'] = nyt_df.apply(lambda x: calculate_word_count(x['cl_headline']), axis=1)
    nyt_df['word_count_abstract'] = nyt_df.apply(lambda x: calculate_word_count(x['cl_abstract']), axis=1)

    # export to interim folder
    if not os.path.exists(os.path.join(INTERIM_DIR, 'nyt_data')):
        os.mkdir(os.path.join(INTERIM_DIR, 'nyt_data'))
    # create filename from input file name
    file_name = os.path.basename(file)
    # save to subdir in interim data folder 
    nyt_df.to_csv(os.path.join(INTERIM_DIR, f"nyt_data/{file_name}"), index=False)


#############################################################
#                   GUARDIAN Dataset
#############################################################