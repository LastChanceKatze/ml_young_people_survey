from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import preprocess as pp
import visualizing_clusters as vis


def elbow_method(data, num_iters):
    scores = []
    for i in range(num_iters):
        km = KMeans(n_clusters=i+2, init="k-means++",
                    max_iter=3000, random_state=0)
        km.fit(data)
        scores.append(km.inertia_)

    sns.lineplot(data=scores)
    plt.title("Elbow method")
    plt.xlabel("No. clusters")
    plt.ylabel("Inertia")
    plt.savefig("./graphs/elbow_method_cosine.png")
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

data_movie = data_movie.drop(columns=["Number of siblings", "Only child", "Village - town", "Age"])

# one hot encoding
data_movie = pp.one_hot_encoding(data_movie, start_idx=12)
######################################################################
data_cols = data_movie.columns
data_movie_norm = pp.normalize(data_movie)
# elbow_method(data_movie_norm, 12)

# use 4 clusters for analysis
num_clusters = 4
km = KMeans(n_clusters=num_clusters, init="k-means++",
            max_iter=3000, random_state=0)
km.fit(data_movie_norm)

# evaluate clusters
# silhouette score
print("Silhouette score:", silhouette_score(data_movie_norm, km.labels_, metric='cosine'))

# PCA variance
# vis.pca_selection(data_movie_norm.shape[1], data_movie_norm)

# visualize with PCA
vis.plot_clusters_pca(3, data_movie_norm, km.labels_, num_clusters=num_clusters)

# dataframe with predictions
df_clusters = data_movie.copy()
# df_clusters = pd.DataFrame(data_movie_norm, columns=data_cols)
df_clusters['cluster'] = km.labels_

df_clusters_mean = df_clusters.groupby('cluster').mean() - data_movie.median()
print(df_clusters_mean)

# visualise cluster means
vis.plot_cluster_distribution(df_clusters_mean)

# # visualise cluster centers
# df_centers = pd.DataFrame(km.cluster_centers_, columns=data_cols)
# df_centers = df_centers.drop(["Age"], axis=1)
# vis.plot_cluster_centers(df_centers.iloc[:, :13])
