from sklearn.cluster import AgglomerativeClustering
from sklearn import datasets
import numpy as np
import pandas as pd
import preprocess as pp
from sklearn.metrics import silhouette_score

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
data_movie = pp.preprocess_data(data_movie)
print(data_movie)
clustering = AgglomerativeClustering().fit(data_movie)
print(clustering.labels_)
score = silhouette_score(data_movie, clustering.labels_, metric='euclidean')
#
# Print the score
#
print('Silhouetter Score: %.3f' % score)