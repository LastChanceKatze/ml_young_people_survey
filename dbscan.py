from sklearn.cluster import DBSCAN
import numpy as np
import
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
                 "Gender", "Education",
                   "House - block of flats"]

data_movie = data[movie_columns]

X = np.array([[1, 2], [2, 2], [2, 3],
         [8, 7], [8, 8], [25, 80]])

clustering = DBSCAN(eps=3, min_samples=2).fit(X)
clustering.labels_
DBSCAN(eps=3, min_samples=2)