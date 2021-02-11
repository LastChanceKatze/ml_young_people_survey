from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import pandas as pd
import preprocess as pp
import visualizing_clusters as vis
import evaluate as ev


def elbow_method(data, num_iters):
    scores = []
    n_clusters = []
    for i in range(2, num_iters):
        km = KMeans(n_clusters=i, init="k-means++",
                    max_iter=3000, random_state=0)
        km.fit(data)
        scores.append(km.inertia_)
        n_clusters.append(i)

    plt.plot(n_clusters, scores)
    plt.xticks(n_clusters)
    plt.title("Elbow method")
    plt.xlabel("No. clusters")
    plt.ylabel("Inertia")
    plt.savefig("./graphs/elbow_method_cosine.png")
    plt.show()


def plot_scores(data, num_iters):
    sil_scores = []
    c_h_scores = []
    n_clusters = []
    for i in range(2, num_iters):
        km = KMeans(n_clusters=i, init="k-means++",
                    max_iter=3000, random_state=0)
        km.fit(data)
        sil_scores.append(silhouette_score(data, km.labels_, metric='cosine'))
        c_h_scores.append(calinski_harabasz_score(data, km.labels_))
        n_clusters.append(i)

    fig, axs = plt.subplots(1, 2)

    axs[0].plot(n_clusters, sil_scores)
    axs[0].set(xlabel="No. clusters", ylabel="Silhouette score",
               xticks=n_clusters)

    axs[1].plot(n_clusters, c_h_scores)
    axs[1].set(xlabel="No. clusters", ylabel="Calinski Harabasz score",
               xticks=n_clusters)

    plt.savefig("./graphs/kmeans_scores.png")
    plt.show()


# load data
data = pd.read_csv("dataset/responses.csv")

# select movie columns
select_cols = ["Movies", "Horror", "Thriller", "Comedy", "Romantic",
               "Sci-fi", "War", "Fantasy/Fairy tales", "Animated",
               "Documentary", "Western", "Action", "Age",
               "Number of siblings", "Gender", "Education",
               "Only child", "Village - town", "House - block of flats"]

data_movie = data[select_cols]

# impute data
data_movie = pp.impute(data_movie)

data_movie = data_movie.drop(columns=["Number of siblings", "Only child", "Village - town"])

# one hot encoding
data_movie = pp.one_hot_encoding(data_movie, start_idx=13)
data_cols = data_movie.columns
data_movie_norm = pp.normalize(data_movie)
######################################################################
# PCA variance selection
# data_pca, var_ratio = pp.pca_selection(data_movie_norm.shape[1], data_movie_norm)
# pp.plot_pca(variance=np.cumsum(var_ratio))

# pca data with x components
data_pca, var_ratio = pp.pca_selection(15, data_movie_norm)
data_pca_norm = pp.normalize(data_pca)

elbow_method(data_pca_norm, 20)
plot_scores(data_pca_norm, 10)

# use 4 clusters for analysis
num_clusters = 3
km = KMeans(n_clusters=num_clusters, init="k-means++",
            max_iter=3000, random_state=0)
km.fit(data_movie_norm)

# evaluate clusters
# silhouette score
ev.eval_scores(data_movie_norm, km.labels_)

# visualize with PCA 2D
# vis.plot_clusters_pca_2d(3, data_movie_norm, km.labels_, num_clusters=num_clusters)
# vis.plot_clusters_pca_3d(3, data_movie_norm, km.labels_, num_clusters=num_clusters)

# dataframe with predictions
df_clusters = data_movie.copy()
# df_clusters = pd.DataFrame(data_movie_norm, columns=data_cols)
df_clusters['cluster'] = km.labels_

df_clusters_mean = df_clusters.groupby('cluster').mean() - data_movie.median()
print(df_clusters_mean)

# visualise cluster means
# vis.plot_cluster_distribution(df_clusters_mean)

vis.plot_count_cluster(df_clusters['cluster'], data_movie['Gender_male'], col_name="Gender")
