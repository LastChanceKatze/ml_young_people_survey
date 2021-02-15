from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import pandas as pd
import descriptive_statistics as ds


def plot_clusters_pca_2d(n_components, data, labels, num_clusters):
    """
    Transform the data using PCA.
    Then plot the clusters using the first 2 components
    """
    pca = PCA(n_components=n_components)
    x_pca = pca.fit_transform(data)

    sns.scatterplot(x=x_pca[:, 0], y=x_pca[:, 1], hue=labels)\
        .set_title(f"PCA: {num_clusters} clusters")

    plt.savefig(f"./graphs/pca_{num_clusters}_2D.png")
    plt.show()


def plot_clusters_pca_3d(n_components, data, labels, num_clusters):
    """
    Transform the data using PCA.
    Then plot the clusters using the first 2 components
    """
    pca = PCA(n_components=n_components)
    x_pca = pca.fit_transform(data)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(xs=x_pca[:, 0], ys=x_pca[:, 1], zs=x_pca[:, 2], c=labels)

    plt.savefig(f"./graphs/pca_{num_clusters}_3D.png")
    plt.show()


def plot_cluster_distribution(df_cluster, num_clusters):
    fig, axs = plt.subplots(1, num_clusters)

    for i in range(num_clusters):
        plt.figure(figsize=(10, 8))

        df_cluster.iloc[i, :].plot(kind='bar', ax=axs[i])

    plt.show()


def plot_count_cluster(labels, data_column, col_name):
    plt.subplots(figsize=(15, 5))
    sns.countplot(x=labels, hue=data_column)
    plt.savefig(f"./graphs/k_means_count_{col_name}.png")
    plt.show()


def plot_score(algorithm, data, sil_scores, c_h_scores, nn_metric):
    algorithm.fit(data)
    sil_scores.append(silhouette_score(data, algorithm.labels_, metric=nn_metric))
    c_h_scores.append(calinski_harabasz_score(data, algorithm.labels_))

    return sil_scores, c_h_scores


def plot_scores(data, num_iters, algorithm_name, figure_path, nn_metric):
    sil_scores = []
    c_h_scores = []
    n_clusters = []
    for i in range(2, num_iters):

        if algorithm_name == "kmeans":
            algorithm = KMeans(n_clusters=i, init="k-means++",
                               max_iter=3000, random_state=0)
        elif algorithm_name == "agglomerative":
            algorithm = AgglomerativeClustering(n_clusters=i)

        sil_scores, c_h_scores = plot_score(algorithm, data, sil_scores, c_h_scores, nn_metric)
        n_clusters.append(i)

    fig, axs = plt.subplots(1, 2)

    axs[0].plot(n_clusters, sil_scores)
    axs[0].set(xlabel="No. clusters", ylabel="Silhouette score",
               xticks=n_clusters)

    axs[1].plot(n_clusters, c_h_scores)
    axs[1].set(xlabel="No. clusters", ylabel="Calinski Harabasz score",
               xticks=n_clusters)

    plt.savefig(figure_path)
    plt.show()


def plot_count_clusters(data, labels):
    df_group = data.groupby(labels)
    cl_0 = df_group.get_group(2)
    print(cl_0.head(2))
    ds.plot_countplots(cl_0)


