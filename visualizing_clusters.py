from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


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


def plot_cluster_distribution(df_cluster):
    for i in range(df_cluster.shape[0]):
        plt.figure(figsize=(10, 8))

        df_cluster.iloc[i, :].plot(kind='bar')
        plt.title("Distribution of Interests in Cluster %d" % i)
        plt.savefig(f"./graphs/k_means_dist_{i}.png")

        plt.show()


def plot_count_cluster(labels, data_column, col_name):
    plt.subplots(figsize=(15, 5))
    sns.countplot(x=labels, hue=data_column)
    plt.savefig(f"./graphs/k_means_count_{col_name}.png")
    plt.show()
