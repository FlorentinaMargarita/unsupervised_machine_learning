import string
import json
import re
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import mixture
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
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

# read the JSON data file and yield the relevant columns for each line (metadata record).
def load_json_data(path, n):
    with open(path, 'rb') as json_file:
        # for each line in json_file, line number "i"
        for i, line in enumerate(json_file):
            # it was only asked to read n lines, so if it would have read more, it should stop.
            if i >= n: return
            # if it is at a multiple of 50k lines, it prints a progress message with the number of lines read and what percent of
            # the total number of records it is.
            if i % 50000 == 0: print(f'{100*i/1955945.0:.0f}% line {i}')
            # parse the line as JSON.  "paper" is now a dict with the paper metadata.
            paper = json.loads(line)
            # each metadata entry has a list of versions: when it was first
            # uploaded, and then version entries for any subsequent changes.
            # I use the year of the first version (upload) as the date of the paper.
            # the "created" field has a full timestamp, e.g.
            # Tue, 24 Jul 2007 20:10:27 GMT
            # so I find the year by looking for any four digits.
            year = int(re.search(r'\d{4}', paper['versions'][0]['created']).group())
            # clean (handle non-ASCII characters, punctuation, and whitespace)
            # the title and abstract, and yield all the fields of interest.
            yield year, clean(paper['title']), paper['categories'], clean(paper['abstract'])

# call load_json_data to get the fields we'll use from the JSON data file,
# then store them in a dataframe and encode the categories as binary columns.
def load_data(data_path, n):
    # load the file data into a dataframe.
    initial_df = pd.DataFrame(load_json_data(data_path, n), columns=['year', 'title', 'categories', 'abstract'])
    # initial_df.categories has values like "math.CO cs.CG".
    # .str.get_dummies(' ') splits the categories column on space to get the
    # individual category values, then encodes those values as binary columns.
    # for example, "math.CO cs.CG" is split into "math.CO" and "cs.CG", and then
    # encoded (using CATEGORY_DUMMIES_PREFIX) as "has_category_math.CO" and
    # "has_category_cs.CG".  this is done for every row, so there will a column
    # for every unique category value in the data.
    dummies_df = initial_df.categories.str.get_dummies(' ').add_prefix(CATEGORY_DUMMIES_PREFIX)
    # dummies_df only has the category binary columns, not any of the other data
    # like year, title, or abstract, so paste them together and store in
    # df_categories.
    df_categories = pd.concat([initial_df, dummies_df], axis=1)
    # again split the categories column on space, but this time don't encode as binary columns, 
    # simply store the first category in a column.
    # I planned to use this single category column as labels for training data
    # to help identify better TFIDF features, but I didn't finish it.
    df_categories['first_category'] = df_categories.categories.str.split(' ', 1).str[0]
    return df_categories

def main():
    # the path to the JSON data file.
    json_path = 'arxiv-metadata-oai-snapshot.json'
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
    result = run_one(data, params)
    # show the number of papers in and the most significant words for each cluster.
    show_cluster_word_info(result)
    return result

# Below I am building a list of stop words. I start with text.ENGLISH_STOP_WORDS, 
# the English stop words provided by sklearn, and then add some scientific research stop words as well as some numeric strings for years etc.
STOP_WORDS = text.ENGLISH_STOP_WORDS.union(
    ['et', 'al', 'etal', 'phys', 'rev', 'lett', 'comment', 'review', 'using', 'uses', 'used', 'use',
     'new', 'method', 'proposed', 'results', 'based', 'paper', 'problem', 'bf',
     'data', 'analysis', 'experimental', 'work', 'different', 'approach', 'recent', 'given']
    + [str(i) for i in range(1900, 2025)]
    + [str(i) for i in range(101)])

# when the paper categories (e.g. "math.CO cs.CG") are converted into binary columns (e.g. "has_category_math.CO" "has_category_cs.CG"), 
# I give the column names a prefix so they're easy to identify later. 
CATEGORY_DUMMIES_PREFIX = 'has_category_'

# get the number of papers in each cluster using the results "result".
def get_article_count_per_cluster(result):
    # result['label'] is an array of cluster IDs.  
    # the first entry of result['label'] being 5 means the first paper/abstract was assigned to cluster ID 5.
    # np.unique returns the unique values in an array, so in this case the list of unique cluster IDs.  
    # return_counts=True says to include how many instances of each cluster ID there are; i.e. 
    # how many papers were assigned to each cluster ID.
    cluster_ids, counts = np.unique(result['label'], return_counts=True)
    # sort counts descending and save the corresponding indices in sorted_count_idx. 
    sorted_count_idx = np.argsort(-counts)
    # with the sorted indices, get the cluster IDs and cluster counts that correspond to those indices.  
    # that is, get the cluster IDs in descending order of cluster size, and the cluster sizes themselves in descending order. 
    # then pair them up with zip.
    return list(zip(cluster_ids[sorted_count_idx], counts[sorted_count_idx]))

# get the n most significant words in each cluster using the results "result".
def get_top_cluster_words(result, cluster, n=10):
    # result['tfidf_mat'] is a matrix where each row is an abstract and each column is a term.  
    # the values are the TFIDF value for the given term in the given abstract.
    # get each index of result['label'] where the value is the given cluster ID.
    cluster_idx = np.where(result['label'] == cluster)[0]
    # get the full rows of tfidf_mat for only the papers in the given cluster.
    matrix_rows = result['tfidf_mat'][cluster_idx,:]
    # find the column sum for each column.  that is, for each term, find the
    # sum of the TFIDF values over all the papers in this cluster.
    tfs = matrix_rows.sum(0).A1
    # get the indices of the column sums that are nonzero, which means remove the terms that never appear in the cluster.
    nonzero_tfs_idx = tfs.nonzero()
    # get the actual column sums that are nonzero.
    nonzero_tfs = tfs[nonzero_tfs_idx]
    # get the indices of nonzero_tfs sorted descending.
    sorted_nonzero_tfs_idx = np.argsort(-nonzero_tfs)
    feature_names_array = np.array(result['feature_names'])
    # list of term names that appear in the cluster.
    nonzero_feature_names = feature_names_array[nonzero_tfs_idx]
    # term names that appear in the cluster, sorted by total TFIDF descending.
    nonzero_feature_names_tf_desc = nonzero_feature_names[sorted_nonzero_tfs_idx]
    # get pairs of (term name, sum of TFIDF for term across cluster) sorted
    # by sum of TFIDF descending.  this is how we define "top words" in a cluster.
    return list(zip(nonzero_feature_names_tf_desc,
             nonzero_tfs[sorted_nonzero_tfs_idx]))[:n]

# show the number of papers in and the most significant words for each cluster.
def show_cluster_word_info(result):
    for cluster_id, article_count in get_article_count_per_cluster(result):
        cluster_words = get_top_cluster_words(result, cluster_id, 10)
        print(f'{cluster_id}: {article_count} {", ".join(w for w,_ in cluster_words)}')

# show the title of each paper in a given cluster.
def show_titles_for_cluster(result, cluster):
    with pd.option_context('display.max_colwidth', None, 'display.max_rows', None):
        print(result['df'].iloc[np.where(result['label'] == cluster)].title)

# run the ML algorithms on the data and return the results (including labels).
def run_one(data, params):
    n_samples = params['n_samples']
    max_features = params['max_features']
    min_df = params['min_df']
    max_df = params['max_df']
    ngram_max = params['ngram_max']
    n_clusters = params['n_clusters']
    n_init = params['n_init']

    # get the abstracts out.
    abstracts = data.abstract
    # set up the TFIDF vectorizer using our custom stop words.
    # ngram_range=(1, ngram_max) means using each group of N consecutive words as a term, N from 1 to ngram_max.
    # max_features - take only the top max_features terms by frequency. this is set to limit the number of features to keep CPU/memory reasonable.
    # min_df - ignore terms that appear in less than min_df proportion of documents. these words are too rare to help find patterns in the data.
    # max_df - ignore terms that appear in more than max_df proportion of documents.
    # these words are too common, also known as corpus-specific stop words.
    TFIDF = TfidfVectorizer(stop_words=STOP_WORDS, ngram_range=(1,ngram_max), max_features=max_features, min_df=int(min_df) if min_df==int(min_df) else float(min_df), max_df=float(max_df))
    # compute TFIDF values for the abstracts.
    TFIDFtext = TFIDF.fit_transform(abstracts)
    # sklearn recommends a batch size of 256 * core count
    batch_size = 256 * os.cpu_count()
    # I use MiniBatchKMeans because it's faster than KMeans.
    # n_init - KMeans and MiniBatchKMeans start with random centroids that affect the algorithm result.  
    # it will try n_init different ones and use the best.
    label = MiniBatchKMeans(batch_size=batch_size, n_clusters=n_clusters, n_init=n_init, random_state=0).fit_predict(TFIDFtext)
    return dict(df=data, feature_names=TFIDF.get_feature_names(), tfidf_mat=TFIDFtext, label=label)

if __name__ == '__main__':
    result = main()
