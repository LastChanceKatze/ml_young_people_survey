
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import pandas as pd
import preprocess as pp
import visualizing_clusters as vis
import evaluate as ev
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth


def ms_detect_outliers(labels, labels_unique, labels_count, min_points):
    outlier_labels = []
    for label, label_count in zip(labels_unique, labels_count):
        if label_count < min_points:
            outlier_labels.append(label)

    outlier_inx = []
    for i in range(len(labels)):
        if labels[i] in outlier_labels:
            outlier_inx.append(i)

    return outlier_inx

# load data
data = pd.read_csv("dataset/responses.csv")

# select movie columns
select_cols = ["Movies", "Horror", "Thriller", "Comedy", "Romantic",
               "Sci-fi", "War", "Fantasy/Fairy tales", "Animated",
               "Documentary", "Western", "Action", "Age",
               "Number of siblings", "Gender", "Education",
               "Only child", "Village - town", "House - block of flats"]

data_movie = data[select_cols]

data_movie_norm, data_movie, data_cols=pp.preprocess_data(data_movie)

"""
######################################################################
# PCA variance selection
data_pca, var_ratio = pp.pca_selection(data_movie_norm.shape[1], data_movie_norm)
pp.plot_pca(variance=np.cumsum(var_ratio))

# pca data with x components
data_pca, var_ratio = pp.pca_selection(15, data_movie_norm)
data_pca_norm = pp.normalize(data_pca)
#######################################################################
"""
#vis.plot_scores(data_movie_norm, 20, "meanshift", "./graphs/meanshift_scores.png")

bandwidth = estimate_bandwidth(data_movie_norm, quantile=0.1)

alg =  MeanShift(bandwidth=bandwidth, bin_seeding=True)
alg = alg.fit(data_movie_norm)

labels = alg.labels_
cluster_centers = alg.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)


# evaluate clusters
ev.eval_scores(data_movie_norm, alg.labels_)

# visualize with PCA 2D
vis.plot_clusters_pca_2d(3, data_movie_norm, alg.labels_, num_clusters=n_clusters_)
vis.plot_clusters_pca_3d(3, data_movie_norm, alg.labels_, num_clusters=n_clusters_)

# dataframe with predictions
#df_clusters = pd.DataFrame(data_movie_norm, columns=data_cols)
#df_clusters['cluster'] = alg.labels_

#df_clusters_mean = df_clusters.groupby('cluster').mean() - data_movie.median()
#print(df_clusters_mean)

# visualise cluster means
#vis.plot_cluster_distribution(df_clusters_mean, num_clusters)
#vis.plot_count_cluster(df_clusters['cluster'], data_movie['Gender_male'], col_name="Gender")

# plot the top three levels of the dendrogram