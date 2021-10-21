import itertools
import os

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans

from fit import load_data, STOP_WORDS

def run_and_plot(data):
    n_samples = data.shape[0]
    abstracts = data.abstract
    n_init = 10
    ngram_min = 1
    ngram_max = 2
    max_df = 0.8
    ranges = dict(
        n_clusters=range(10,101,5),
        min_df=range(1,21),
        max_features=range(100,1001,50),
    )

    default_params = dict(
        n_clusters=50,
        min_df=2,
        max_features=500
    )
    features = ['n_clusters', 'min_df', 'max_features']
    feature_combination_i = 0
    n_feature_value_combinations = 0
    for feature1, feature2 in itertools.combinations(features, 2):
        n_feature_value_combinations += len(ranges[feature1]) * len(ranges[feature2])
    for feature1, feature2 in itertools.combinations(features, 2):
        xs = []
        ys = []
        zs = []
        remaining_features = set(ranges.keys()) - {feature1, feature2}
    for feature1, feature2 in itertools.combinations(features, 2):
        for value1 in ranges[feature1]:
            for value2 in ranges[feature2]:
                feature_combination_i += 1
                params = default_params.copy()
                params.update({
                    feature1: value1,
                    feature2: value2,
                })
                n_clusters = params['n_clusters']
                min_df = params['min_df']
                max_features = params['max_features']

                TFIDF = TfidfVectorizer(stop_words=STOP_WORDS, ngram_range=(ngram_min,ngram_max), max_features=max_features, min_df=min_df, max_df=max_df)
                TFIDFtext = TFIDF.fit_transform(abstracts)
                n_features = len(TFIDF.vocabulary_)
                if n_features >= n_samples: continue
                tfidf_df = pd.DataFrame(data=TFIDFtext.todense(), columns=TFIDF.get_feature_names())
                batch_size = 256 * os.cpu_count()
                label = MiniBatchKMeans(batch_size=batch_size, n_clusters=n_clusters, n_init=n_init, random_state=0).fit_predict(tfidf_df)

        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
        ax.scatter3D(xs, ys, zs, c=zs, cmap='winter')
        ax.set_xlabel(feature1)
        ax.set_ylabel(feature2)
        ax.set_zlim(bottom=0)
        ax.set_zlabel('Silhouette Score')
        ax.set_title(', '.join(f'{f}={params[f]}' for f in remaining_features))
    plt.show()

if __name__ == '__main__':
    json_path = 'arxiv-metadata-oai-snapshot.json'
    data = load_data(json_path, 2000)
    run_and_plot(data)
