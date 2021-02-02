import pandas as pd
import preprocess as pp

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
print(data_movie.info())

# check missing vals
# data contains missing vals
pp.check_missing_vals(data_movie)

# impute data
data_movie = pp.impute(data_movie)
pp.check_missing_vals(data_movie)

# describe data
print(data_movie.describe())
