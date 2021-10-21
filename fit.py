
import string
import json
import re
import os

from timeit import default_timer as timer

import pandas as pd

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans

def clean(s):
    # remove non-ASCII characters
    s = s.encode('ascii', errors='ignore').decode()
    # remove punctuation
    s = s.translate(str.maketrans('', '', string.punctuation))
    # convert tabs and newlines to spaces
    s = re.sub('[\n\t]', ' ', s)
    # remove leading spaces
    s = re.sub('^\s+', '', s)
    # remove trailing spaces
    s = re.sub('\s+$', '', s)
    # squeeze multiple spaces into one
    s = re.sub(r'\s\s+', ' ', s)
    return s


# read the JSON data file and yield the relevant columns
# for each line (metadata record).
def load_json_data(path, n):
    with open(path, 'rb') as json_file:
        # for each line in json_file, line number "i"
        for i, line in enumerate(json_file):
            # we were only asked to read n lines, so if we've read more,
            # we should stop.
            if i >= n: return
            # if I'm at a multiple of 50k lines, let's print a progress
            # message with the number of lines read and what percent of
            # the total number of records it is.
            if i % 50000 == 0: print(f'{100*i/1955945.0:.0f}% line {i}')
            # parse the line as JSON.  "paper" is now a dict with the
            # paper metadata.
            paper = json.loads(line)
            # each metadata entry has a list of versions: when it was first
            # uploaded, and then version entries for any subsequent changes.
            # I use the year of the first version (upload) as the date
            # of the paper.
            # the "created" field has a full timestamp, e.g.
            # Tue, 24 Jul 2007 20:10:27 GMT
            # so I find the year by looking for any four digits.
            year = int(re.search(r'\d{4}', paper['versions'][0]['created']).group())
            # clean (handle non-ASCII characters, punctuation, and whitespace)
            # the title and abstract, and yield all the fields of interest.
            yield year, clean(paper['title']), paper['categories'], clean(paper['abstract'])

# for row in load_json_data(r'C:\Users\flore\Downloads\WPy64-3950\notebooks\arxiv-metadata-oai-snapshot.json', 5):
#     print(row) 


# Here I build our list of stop words. I start with text.ENGLISH_STOP_WORDS, the EnglisH stop words provided by sklearn, and then I add some scientific research
# stop words as well as some numeric strings.
STOP_WORDS = text.ENGLISH_STOP_WORDS.union(
    ['et', 'al', 'etal', 'phys', 'rev', 'lett', 'comment', 'review', 'using', 'uses', 'used', 'use',
     'new', 'method', 'proposed', 'results', 'based', 'paper', 'problem', 'bf',
     'data', 'analysis', 'experimental', 'work', 'different', 'approach', 'recent', 'given']
    # getting rid of any page numbers as well as any year.
    + [str(i) for i in range(1900, 2025)]
    + [str(i) for i in range(101)])



# when I convert the paper categories (e.g. "math.CO cs.CG") into binary
# columns (e.g. "has_category_math.CO" "has_category_cs.CG"), I'd like to
# give the column names a prefix so they're easy to identify later.  this is
# that prefix.
CATEGORY_DUMMIES_PREFIX = 'has_category_'

    
def load_data(data_path, n):
    # initial_df.categories has values like "math.CO cs.CG".
    # .str.get_dummies(' ') splits the categories column on the white space to get the
    # individual category values, then encodes those values as binary columns.
    # for example, "math.CO cs.CG" is split into "math.CO" and "cs.CG", and then
    # encoded (using CATEGORY_DUMMIES_PREFIX) as "has_category_math.CO" and
    # "has_category_cs.CG".  this is done for every row, so there will a column
    # for every unique category value in the data.
    initial_df = pd.DataFrame(load_json_data(data_path, n), columns=['year', 'title', 'categories', 'abstract'])
    # dummies_df only has the category binary columns, not any of the other data
    # like year, title, or abstract, so I paste them together and store in df_categories.
    dummies_df = initial_df.categories.str.get_dummies(' ').add_prefix(CATEGORY_DUMMIES_PREFIX)
    #  axis=1 stands for columns.  axis=0 would be rows. Here new columns are added.
    df_categories = pd.concat([initial_df, dummies_df], axis=1)
    return df_categories


def main():
    # the path to the JSON data file.
    json_path = '../../../Arxiv ML/arxiv-metadata-oai-snapshot.json'
    # parameters to ML algorithms, mostly for TFIDF.
    params = dict(
        max_features=10000,
        min_df=5,
        max_df=0.8,
        ngram_max=2,
        n_clusters=600,
        n_samples=100000,
        n_init=3
    )
    # read the data (year, title, categories, abstract) from the JSON data file.
    data = load_data(json_path, 100000)
    # run the ML algorithms on that data using the specified parameters.
    # TFIDF and k-means
    result = run_one(data, params)
    # show the number of papers in and the most significant words for
    # each cluster.
    return result


# run the ML algorithms on the data and return the results (including labels).
def run_one(data, params):
    max_features = params['max_features']
    min_df = params['min_df']
    max_df = params['max_df']
    ngram_max = params['ngram_max']
    n_clusters = params['n_clusters']
    n_init = params['n_init']

    # get the abstracts out.
    abstracts = data.abstract
    # set up the TFIDF vectorizer using our custom stop words.
    # ngram_range=(1, ngram_max) means we're using each group of N consecutive
    # words as a term, N from 1 to ngram_max.
    # max_features - take only the top max_features terms by frequency.  we set this
    #   to limit the number of features to keep CPU/memory reasonable.
    # min_df - ignore terms that appear in less than min_df proportion of documents.
    #   these words are too rare to help find patterns in the data.
    # max_df - ignore terms that appear in more than max_df proportion of documents.
    #   these words are too common, also known as corpus-specific stop words.
    TFIDF = TfidfVectorizer(stop_words=STOP_WORDS, ngram_range=(1,ngram_max), max_features=max_features, min_df=int(min_df) if min_df==int(min_df) else float(min_df), max_df=float(max_df))
    # compute TFIDF values for the abstracts.
    TFIDFtext = TFIDF.fit_transform(abstracts)
    # sklearn recommends a batch size of 256 * core count
    batch_size = 256 * os.cpu_count()
    # use MiniBatchKMeans because it's faster than KMeans.
    # n_init - KMeans and MiniBatchKMeans start with random centroids that
    #   affect the algorithm result.  it will try n_init different ones and use
    #   the best.
    label = MiniBatchKMeans(batch_size=batch_size, n_clusters=n_clusters, n_init=n_init, random_state=0).fit_predict(TFIDFtext)
    return dict(df=data, feature_names=TFIDF.get_feature_names(), tfidf_mat=TFIDFtext, label=label)

if __name__ == '__main__':
    result = main()