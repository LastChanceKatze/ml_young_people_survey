from sklearn.cluster import DBSCAN
from sklearn import datasets
import numpy as np
import pandas as pd
import preprocess as pp
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import visualizing_clusters as vis
import numpy as np
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as shc



sns.set()

def find_optimal_eps(data, nn_metric):
    neigh = NearestNeighbors(n_neighbors=2, metric=nn_metric)
    nbrs = neigh.fit(data)
    distances, indices = nbrs.kneighbors(data)

    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    plt.plot(distances)
    plt.show()

def dbscan_start(data, nn_metric):

    #Determine centroids
    #centers = [[0.5, 2], [-1, -1], [1.5, -1]]

    #Create dataset
    #data, y = make_blobs(n_samples=400, centers=centers, 
     #             cluster_std=0.5, random_state=0)
    #Normalize the values
    data, var_ratio = pp.pca_selection(15, data)
    data=pp.scale_features(data)
    find_optimal_eps(data, nn_metric)

    clustering = DBSCAN(eps=0.32, min_samples=32, metric=nn_metric).fit(data)
    #print(clustering.labels_)
  #  score = silhouette_score(data, clustering.labels_, metric=nn_metric)
    #score_calinski = calinski_harabasz_score(data, clustering.labels_)

    #
    # Print the score
    #
   # print('Silhouetter Score: %.3f' % score) 
   # print('Calinski Score: %.3f' % score_calinski) 
    labels=clustering.fit_predict(data)
    """"
    y_pred = clustering.fit_predict(data)
    plt.figure(figsize=(10,6))
    plt.scatter(data[:,0], data[:,1],c=y_pred, cmap='Paired')
    plt.title("Clusters determined by DBSCAN")
    plt.show()
    """

    n_clusters_ = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
    n_noise_ = list(clustering.labels_).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

    vis.plot_clusters_pca_2d(3, data, labels, num_clusters=n_clusters_)
    vis.plot_clusters_pca_3d(3, data, labels, num_clusters=n_clusters_)

  

desired_width = 500
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 20)

# load data
data = pd.read_csv("dataset/responses.csv")
print(data.head(2))

# select movie columns
movie_columns = ["Movies", "Horror", "Thriller", "Comedy", "Romantic",
                 "Sci-fi", "War", "Fantasy/Fairy tales", "Animated",
                 "Documentary", "Western", "Action", "Age",
                 "Number of siblings", "Gender", "Education",
                 "Only child", "Village - town", "House - block of flats"]

data_movie = data[movie_columns]
data_movie_norm, data_movie, data_cols = pp.preprocess_data(data_movie)

dbscan_start(data_movie, 'cosine')




