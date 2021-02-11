from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def plot_pca(variance):
    plt.plot(variance)
    plt.xlabel('Number of components')
    plt.ylabel("Variance")
    plt.show()


def pca_selection(n_components, data):
    pca = PCA(n_components=n_components, random_state=200, whiten=True)
    pca.fit_transform(data)

    var_ratio = pca.explained_variance_ratio_*100
    print("Variance ratio - PCA\n", var_ratio)
    print("Total variance - PCA: ", np.sum(var_ratio))
    plot_pca(variance=np.cumsum(var_ratio))


def plot_clusters_pca(n_components, data, labels, num_clusters):
    """
    Transform the data using PCA.
    Then plot the clusters using the first 2 components
    """
    pca = PCA(n_components=n_components)
    x_pca = pca.fit_transform(data)

    sns.scatterplot(x=x_pca[:, 0], y=x_pca[:, 1], hue=labels)\
        .set_title(f"PCA: {num_clusters} clusters")

    plt.savefig(f"./graphs/pca_{num_clusters}.png")
    plt.show()


def plot_cluster_distribution(df_cluster):
    for i in range(df_cluster.shape[0]):
        plt.figure(figsize=(10, 8))

        df_cluster.iloc[i, :].plot(kind='bar')
        plt.title("Distribution of Interests in Cluster %d" % i)
        plt.savefig(f"./graphs/k_means_dist_{i}.png")

        plt.show()


def plot_cluster_centers(df_center):
    """
    Plot the centers of the clusters on a line plot
    """
    cols = df_center.columns
    df_center = df_center.T
    df_center.plot.line(xticks=range(len(cols)))
    plt.savefig("./graphs/k_means_centers.png")
    plt.show()

