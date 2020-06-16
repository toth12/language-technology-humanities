"""Creates a kmeans clustering of document term matrix and plot it with tsney."""
import json
import os
import pdb
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import TSNE
import matplotlib.cm as cm
from matplotlib import colors as mcolors
from sklearn.decomposition import PCA


# Add current path to python path
sys.path.append(os.getcwd())
import constants


# Read the input data
input_directory = os.getcwd() + '/' + constants.OUTPUT_FOLDER + \
    'Anglo_Saxon_Chronicles/'
# Load the document term matrix
matrix_documents = np.loadtxt(input_directory + 'document_term_matrix.txt')

# Load all features
with open(input_directory + 'selected_features.json') as json_file:
    features = json.load(json_file)


def find_optimal_clusters(data, max_k):
    iters = range(2, max_k + 1, 2)
    sse = []
    for k in iters:
        sse.append(MiniBatchKMeans(n_clusters=k, init_size=1024,
                                   batch_size=2048,
                                   random_state=20).fit(data).inertia_)
        print('Fit {} clusters'.format(k))
    f, ax = plt.subplots(1, 1)
    ax.plot(iters, sse, marker='o')
    ax.set_xlabel('Cluster Centers')
    ax.set_xticks(iters)
    ax.set_xticklabels(iters)
    ax.set_ylabel('SSE')
    ax.set_title('SSE by Cluster Center Plot')
    plt.savefig(input_directory + 'cluster_k_scores.png')


#find_optimal_clusters(matrix_documents, 50)

clusters = MiniBatchKMeans(n_clusters=12, init_size=1024, batch_size=2048,
                           random_state=20).fit_predict(matrix_documents)


def plot_tsne_pca(data, labels):
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

    # Sort colors by hue, saturation, value and name.
    by_hsv = sorted(
        (tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
        for name, color in colors.items())
    sorted_names = [name for hsv, name in by_hsv]
    my_colours = sorted_names[::3]
    
    groups = set(labels.tolist())
    cluster_colors = []
    for element in groups:
        cluster_colors.append(my_colours[element])

    sample_colors = []
    for element in labels.tolist():
        sample_colors.append(cluster_colors[element])
    max_label = max(labels)
    max_items = np.random.choice(range(data.shape[0]), size=data.shape[0],
                                 replace=False)
    pca = PCA(n_components=2).fit_transform(data[max_items, :])
    tsne = TSNE().fit_transform(
        PCA(n_components=50).fit_transform(data[max_items, :]))
    idx = np.random.choice(range(pca.shape[0]), size=300, replace=False)
    label_subset = labels[max_items]
    label_subset = [cm.hsv(i / max_label) for i in label_subset[idx]]
    f, ax = plt.subplots(1, figsize=(20, 20))
    ax.set_title('TSNE Cluster Plot')

    final_sample_colours = []
    for i in idx:
        final_sample_colours.append(sample_colors[i])
    ax.scatter(tsne[idx, 0], tsne[idx, 1], color=final_sample_colours, s=128)
    plt.savefig(input_directory + 'cluster_plots_without_annotation.png')

    annotations = []
    for i in idx:
        annotations.append(sample_colors[i])

    for i, element in enumerate(tsne):
        if i in idx:
            ax.annotate(i, (element[0], element[1]))
    pdb.set_trace()
    plt.savefig(input_directory + 'cluster_plots_with_annotation.png')


plot_tsne_pca(matrix_documents, clusters)


def get_top_keywords(data, clusters, labels, n_terms):
    df = pd.DataFrame(data).groupby(clusters).mean()

    for i, r in df.iterrows():
        print('\nCluster {}'.format(i))
        print(','.join([labels[t] for t in np.argsort(r)[-n_terms:]]))


get_top_keywords(matrix_documents, clusters, features, 10)

pdb.set_trace()
