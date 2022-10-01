# Collecting and Exploring Datasets

This repo contains code and results of my explorations of text data from different sources. The aim is to create a dataset that consists of related longer and short texts from the same source or plattform.  Longer texts are such as blog posts or news articles (full-text) and short text, should be similar in length to social media posts, such as abstracts or taglines.

**First idea: The New York Times API**  
NYT offers a set of APIs and also have one for their archive. It contains partial articles (e.g. headline, abstract) and other information, such as which section an article belongs to (e.g. Arts, News) and the length of it (word count).  
Unfortunately they only offer unlimited access to the full-texts in the [article archive](https://help.nytimes.com/hc/en-us/articles/115014772767-New-York-Times-Archived-Articles-and-TimesMachine-) for subscribed users. 

Althogh I won't be using their data for my research project, the available data is useful to find out some characteristics of news articles, e.g. the average length or how long their abstracts and titles are. This information could help in finding similar open-access data.

## Repo Structure

```bash
.
├── README.md
├── config.ini
├── data
│   ├── raw
│   └── interim
├── notebooks
│   ├── 01_nyt_apis.ipynb
│   ├── 02_guardian_api.ipynb
│   ├── 03_twitter_api.ipynb
│   └── 04_explore_datasets.ipynb
├── requirements.in
├── requirements.txt
└── src
    ├── __init__.py
    ├── data
    │   ├── __init__.py
    │   ├── clean_text.py
    │   └── make_dataset.py
    └── main.py
```

## Datasets

**NYT Dataset:**  
The dataset contains partial article data from the NYT Archive API from 10/2021 to 09/2022, approx. 34,700 items.

**GU Dataset:**  
The dataset contains article data from the GU Search API from 27/09/2021 to 27/09/2022, approx. 75,000 items.

**GU Tweets Dataset:**  
The dataset contains tweets from The Guardian main account ([@guardian](https://twitter.com/guardian)), with 3,200 items (maximum allowed number of tweets from a user as per Twitter API limitations).


```python
import os
import glob
import pandas as pd

import matplotlib.pyplot as plt

# Notebook settings
import warnings
warnings.filterwarnings('ignore')
```


```python
# Set data directory paths
nyt_path = 'data/interim/nyt_data'
gu_path = 'data/interim/gu_data'
gu_twitter_path = 'data/interim/gu_twitter_data/gu_tweets.csv'
```


```python
# Load NYT data files and combine them
nyt_files = glob.glob(os.path.join(nyt_path, '*.csv'))
nyt_lst = []

for file in nyt_files:
    nyt_single_df = pd.read_csv(file)
    nyt_lst.append(nyt_single_df)

nyt_df = pd.concat(nyt_lst)
```


```python
# Load GU data files and combine them
gu_files = glob.glob(os.path.join(gu_path, '*.csv'))
gu_lst = []

for file in gu_files:
    gu_single_df = pd.read_csv(file)
    gu_lst.append(gu_single_df)

gu_df = pd.concat(gu_lst)
```


```python
# Load GU Twitter data
twitter_df = pd.read_csv(gu_twitter_path)
```

### The New York Times

```python
nyt_df.describe()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>word_count</th>
      <th>word_count_headline</th>
      <th>word_count_abstract</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>34752.000000</td>
      <td>34752.000000</td>
      <td>34752.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>991.256417</td>
      <td>9.483943</td>
      <td>22.524660</td>
    </tr>
    <tr>
      <th>std</th>
      <td>698.379953</td>
      <td>2.946029</td>
      <td>7.412678</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>525.000000</td>
      <td>8.000000</td>
      <td>18.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>934.000000</td>
      <td>10.000000</td>
      <td>23.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1315.000000</td>
      <td>11.000000</td>
      <td>27.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>20573.000000</td>
      <td>22.000000</td>
      <td>103.000000</td>
    </tr>
  </tbody>
</table>
</div>

```python
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(12,5))
fig.suptitle('Word counts for NYT articles')
fig.supylabel('Frequency')

ax1.hist(nyt_df['word_count_headline'], bins=20)
ax1.set_title('Headline')
ax2.hist(nyt_df['word_count_abstract'], bins=100)
ax2.set_title('Abstract')
ax3.hist(nyt_df['word_count'], bins=100)
ax3.set_title('Full-text')

for ax in (ax1, ax2, ax3):
    ax.set(xlabel='Word count')

plt.show()
```

    
![png](README_files/README_13_0.png)
    


### Tha Guardian


```python
gu_df.describe()
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>wordcount</th>
      <th>charCount</th>
      <th>word_count_headline</th>
      <th>word_count_trailText</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>74918.000000</td>
      <td>74918.000000</td>
      <td>74918.000000</td>
      <td>74918.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>779.966110</td>
      <td>4644.092381</td>
      <td>11.642970</td>
      <td>19.905043</td>
    </tr>
    <tr>
      <th>std</th>
      <td>465.097429</td>
      <td>2727.397471</td>
      <td>2.942463</td>
      <td>5.496729</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>498.000000</td>
      <td>2979.000000</td>
      <td>10.000000</td>
      <td>16.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>711.000000</td>
      <td>4239.000000</td>
      <td>12.000000</td>
      <td>19.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>947.000000</td>
      <td>5654.000000</td>
      <td>13.000000</td>
      <td>23.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9633.000000</td>
      <td>54826.000000</td>
      <td>28.000000</td>
      <td>77.000000</td>
    </tr>
  </tbody>
</table>
</div>


```python
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(12,5))
fig.suptitle('Word counts for GUARDIAN articles')
fig.supylabel('Frequency')

ax1.hist(gu_df['word_count_headline'], bins=30)
ax1.set_title('Headline')
ax2.hist(gu_df['word_count_trailText'], bins=100)
ax2.set_title('Abstract')
ax3.hist(gu_df['wordcount'], bins=100)
ax3.set_title('Full-text')

for ax in (ax1, ax2, ax3):
    ax.set(xlabel='Word count')

plt.show()
```
    
![png](README_files/README_16_0.png)
    


### Tweets from The Guardian


```python
twitter_df.head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>created_at</th>
      <th>id</th>
      <th>text</th>
      <th>clean_text</th>
      <th>word_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Wed Sep 28 21:17:47 +0000 2022</td>
      <td>1575233331943309339</td>
      <td>Morning mail: hurricane with 240km/h winds hit...</td>
      <td>Morning mail hurricane with kmh winds hits Flo...</td>
      <td>14</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Wed Sep 28 21:17:46 +0000 2022</td>
      <td>1575233327329574924</td>
      <td>R Kelly ordered to pay restitution of $300,000...</td>
      <td>R Kelly ordered to pay restitution of to his v...</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Wed Sep 28 21:17:44 +0000 2022</td>
      <td>1575233321063292928</td>
      <td>Two aircraft involved in ‘minor collision’ on ...</td>
      <td>Two aircraft involved in ‘minor collision’ on ...</td>
      <td>10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Wed Sep 28 21:10:00 +0000 2022</td>
      <td>1575231372490309658</td>
      <td>We’re keen to hear from people who have recent...</td>
      <td>We’re keen to hear from people who have recent...</td>
      <td>25</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Wed Sep 28 21:03:05 +0000 2022</td>
      <td>1575229631912869894</td>
      <td>Guardian front page, Thursday 29 September 202...</td>
      <td>Guardian front page Thursday September Banks £...</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>




```python
twitter_df.describe()
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>word_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3.200000e+03</td>
      <td>3200.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.572793e+18</td>
      <td>12.177188</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.462916e+15</td>
      <td>3.107159</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.570134e+18</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.571554e+18</td>
      <td>10.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.572913e+18</td>
      <td>12.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.574037e+18</td>
      <td>14.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.575233e+18</td>
      <td>25.000000</td>
    </tr>
  </tbody>
</table>
</div>


```python
plt.hist(twitter_df['word_count'], bins=25)
plt.xlabel('Word count')
plt.ylabel('Frequency')
plt.title('Histogram of Word Counts in GUARDIAN Tweets')
plt.show()
```

    
![png](README_files/README_20_0.png)
    


```python
gu_df.head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>sectionId</th>
      <th>sectionName</th>
      <th>webPublicationDate</th>
      <th>webUrl</th>
      <th>apiUrl</th>
      <th>pillarId</th>
      <th>pillarName</th>
      <th>byline</th>
      <th>body</th>
      <th>...</th>
      <th>lang</th>
      <th>bodyText</th>
      <th>charCount</th>
      <th>bylineHtml</th>
      <th>fields.contributorBio</th>
      <th>scheduledPublicationDate</th>
      <th>cl_headline</th>
      <th>cl_trailText</th>
      <th>word_count_headline</th>
      <th>word_count_trailText</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>sport/blog/2022/jun/27/imperious-nsw-seize-adv...</td>
      <td>sport</td>
      <td>Sport</td>
      <td>2022-06-26T23:46:44Z</td>
      <td>https://www.theguardian.com/sport/blog/2022/ju...</td>
      <td>https://content.guardianapis.com/sport/blog/20...</td>
      <td>pillar/sport</td>
      <td>Sport</td>
      <td>Nick Tedeschi</td>
      <td>&lt;p&gt;A star was born in debutant Matt Burton. An...</td>
      <td>...</td>
      <td>en</td>
      <td>A star was born in debutant Matt Burton. An al...</td>
      <td>4835</td>
      <td>&lt;a href="profile/nick-tedeschi"&gt;Nick Tedeschi&lt;/a&gt;</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Imperious NSW seize advantage after Queensland...</td>
      <td>Poor tackling basic handling errors and a lack...</td>
      <td>10</td>
      <td>23</td>
    </tr>
    <tr>
      <th>1</th>
      <td>music/2022/jun/27/kendrick-lamar-at-glastonbur...</td>
      <td>music</td>
      <td>Music</td>
      <td>2022-06-26T23:32:10Z</td>
      <td>https://www.theguardian.com/music/2022/jun/27/...</td>
      <td>https://content.guardianapis.com/music/2022/ju...</td>
      <td>pillar/arts</td>
      <td>Arts</td>
      <td>Alexis Petridis</td>
      <td>&lt;p&gt;As Glastonbury 2022 draws to a close, a var...</td>
      <td>...</td>
      <td>en</td>
      <td>As Glastonbury 2022 draws to a close, a variat...</td>
      <td>3510</td>
      <td>&lt;a href="profile/alexispetridis"&gt;Alexis Petrid...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Kendrick Lamar at Glastonbury review – faith f...</td>
      <td>Sporting a bejewelled crown of thorns and with...</td>
      <td>11</td>
      <td>25</td>
    </tr>
    <tr>
      <th>2</th>
      <td>world/2022/jun/27/garbage-island-no-more-how-o...</td>
      <td>world</td>
      <td>World news</td>
      <td>2022-06-26T23:06:58Z</td>
      <td>https://www.theguardian.com/world/2022/jun/27/...</td>
      <td>https://content.guardianapis.com/world/2022/ju...</td>
      <td>pillar/news</td>
      <td>News</td>
      <td>Justin McCurry on Teshima island</td>
      <td>&lt;p&gt;Toru Ishii remembers when the shredded car ...</td>
      <td>...</td>
      <td>en</td>
      <td>Toru Ishii remembers when the shredded car tyr...</td>
      <td>6047</td>
      <td>&lt;a href="profile/justinmccurry"&gt;Justin McCurry...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Garbage island no more how one Japanese commun...</td>
      <td>Teshima – site of Japan’s worst case of illega...</td>
      <td>14</td>
      <td>27</td>
    </tr>
    <tr>
      <th>3</th>
      <td>education/2022/jun/27/university-college-londo...</td>
      <td>education</td>
      <td>Education</td>
      <td>2022-06-26T23:01:00Z</td>
      <td>https://www.theguardian.com/education/2022/jun...</td>
      <td>https://content.guardianapis.com/education/202...</td>
      <td>pillar/news</td>
      <td>News</td>
      <td>Richard Adams Education editor</td>
      <td>&lt;p&gt;University College London has boasted its f...</td>
      <td>...</td>
      <td>en</td>
      <td>University College London has boasted its fina...</td>
      <td>4009</td>
      <td>&lt;a href="profile/richardadams"&gt;Richard Adams&lt;/...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>University College London generates £bn a year...</td>
      <td>UCL’s research knowledge and support for busin...</td>
      <td>11</td>
      <td>16</td>
    </tr>
    <tr>
      <th>4</th>
      <td>media/2022/jun/27/young-people-must-report-har...</td>
      <td>media</td>
      <td>Media</td>
      <td>2022-06-26T23:01:00Z</td>
      <td>https://www.theguardian.com/media/2022/jun/27/...</td>
      <td>https://content.guardianapis.com/media/2022/ju...</td>
      <td>pillar/news</td>
      <td>News</td>
      <td>Dan Milmo</td>
      <td>&lt;p&gt;Young people should report harmful online c...</td>
      <td>...</td>
      <td>en</td>
      <td>Young people should report harmful online cont...</td>
      <td>2745</td>
      <td>&lt;a href="profile/danmilmo"&gt;Dan Milmo&lt;/a&gt;</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Young people must report harmful online conten...</td>
      <td>Ofcom says of to yearolds have seen harmful co...</td>
      <td>10</td>
      <td>16</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>

The NYT has more longer articles than the GU. The headlines show the opposite behaviour. The abstracts seem similar in both.  
Compared to the length of tweets, the most similar article part in length is the headline.

## Articles per Category/Section in The Guardian

```python
gu_df['sectionName'].value_counts().nlargest(25).sort_values(ascending=True).plot(kind='barh')
plt.xlabel("Number of Articles", labelpad=14)
plt.ylabel("Section", labelpad=14)
plt.title("Number of Articles per Section in GUARDIAN (top 25 Sections)")
plt.show()
```
    
![png](README_files/README_24_0.png)
