import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'src/data'))

import glob
import json
import re
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
    nyt_df = nyt_df[nyt_df['headline'].notna() & nyt_df['abstract'].notna()]
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
#                   GUARDIAN API Dataset
#############################################################
gu_files = glob.glob(os.path.join(GUARDIAN_DATA_DIR, '*.json'))



# prep with first file
# load first of the files
f = open(gu_files[0])
gu_first_data = json.load(f)
f.close()

gu_first_df = pd.json_normalize(gu_first_data)

# create dictionary to rename columns with prefix
colnames = {}

for colname in gu_first_df.columns:
    if colname.startswith('fields.'):
        colnames[colname] = re.sub(r"fields.", '', colname)

cols_to_drop = [
        'isHosted', 'displayHint', 'firstPublicationDate', 'isInappropriateForSponsorship',
        'isPremoderated', 'lastModified', 'productionOffice', 'shortUrl', 'shouldHideAdverts',
        'showInRelatedContent', 'thumbnail', 'legallySensitive', 'isLive', 'shouldHideReaderRevenue',
        'showAffiliateLinks', 'newspaperPageNumber', 'newspaperEditionDate', 'commentCloseDate', 'commentable', 
        'starRating', 'liveBloggingNow', 'sensitive', 'type', 'main',
        'webTitle', 'standfirst'
    ]

# Process al files in directory
for file in gu_files:
    print(f"Processing file {file} ...")
    f = open(file)
    gu_data = json.load(f)
    f.close()

    # turn json into dataframe with nested structures flattened
    gu_df = pd.json_normalize(gu_data)

    # Rename columns
    gu_df_renamed = gu_df.rename(columns=colnames)

    # select only type == articles
    gu_articles = gu_df_renamed[gu_df_renamed['type'] == 'article']
    gu_articles.drop(cols_to_drop, axis=1, inplace=True)

    # clean up headline and abstract
    gu_articles = gu_articles[gu_articles['headline'].notna() & gu_articles['trailText'].notna() \
        & gu_articles['body'].notna() & gu_articles['bodyText'].notna()]
    gu_articles['cl_headline'] = gu_articles.apply(lambda x: clean_txt(x['headline']), axis=1)
    gu_articles['cl_trailText'] = gu_articles.apply(lambda x: clean_txt(x['trailText']), axis=1)
    gu_articles.drop(['headline', 'trailText'], axis=1, inplace=True)

    # calculate word counts for both columns
    gu_articles['word_count_headline'] = gu_articles.apply(lambda x: calculate_word_count(x['cl_headline']), axis=1)
    gu_articles['word_count_trailText'] = gu_articles.apply(lambda x: calculate_word_count(x['cl_trailText']), axis=1)

    # export to interim folder
    if not os.path.exists(os.path.join(INTERIM_DIR, 'gu_data')):
        os.mkdir(os.path.join(INTERIM_DIR, 'gu_data'))
    # create filename from input file name
    file_name = os.path.splitext(os.path.basename(file))[0]
    # save to subdir in interim data folder 
    gu_articles.to_csv(os.path.join(INTERIM_DIR, f"gu_data/{file_name}.csv"), index=False)

    
#############################################################
#                   GUARDIAN Tweets Dataset
#############################################################
# Load dataset
twitter_df = pd.read_csv(os.path.join(TWITTER_GU_DATA_DIR, 'gu_tweets.csv'))
# replace NaN values with empty strings
twitter_df = twitter_df[twitter_df['text'].notna()]
twitter_df['clean_text'] = twitter_df.apply(lambda x: clean_txt(x['text']), axis=1)
# calculate word counts for both columns
twitter_df['word_count'] = twitter_df.apply(lambda x: calculate_word_count(x['clean_text']), axis=1)

# save cleaned data to interim data folder
if not os.path.exists(os.path.join(INTERIM_DIR, 'gu_twitter_data')):
        os.mkdir(os.path.join(INTERIM_DIR, 'gu_twitter_data'))
twitter_df.to_csv(os.path.join(INTERIM_DIR, 'gu_twitter_data/gu_tweets.csv'), index=False)