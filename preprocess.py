import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, Normalizer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

desired_width = 500
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 20)


# check missing values
def check_missing_vals(data):
    # check for missing values
    missing_vals = data.isna().sum()
    print("Missing values: \n" + str(missing_vals))
    return missing_vals.sum() != 0


def impute(data):
    num_data = data[data.columns[0:14]]
    im_num = SimpleImputer(missing_values=np.nan, strategy="median", copy=False)
    im_num.fit_transform(num_data)

    cat_data = data[data.columns[14:]]
    im_cat = SimpleImputer(missing_values=np.nan, strategy="most_frequent", copy=False)
    im_cat.fit_transform(cat_data)

    return pd.concat([num_data, cat_data], axis=1)


# encode labels
def encode_labels(data):
    data_cp = data.copy()
    cols = data_cp.columns
    encode_quality = LabelEncoder()

    for i in range(14, data_cp.shape[1]):
        data_cp[cols[i]] = encode_quality.fit_transform(data_cp[cols[i]])

    return data_cp


# one hot encoding
def one_hot_encoding(data, start_idx):
    data_cp = data.copy()
    cols = data_cp.columns

    for i in range(start_idx, data_cp.shape[1]):
        df = pd.get_dummies(data=data[cols[i]], prefix=cols[i], drop_first=True)
        data_cp = data_cp.drop(cols[i], axis=1)
        data_cp = pd.concat([data_cp, df], axis=1)

    return data_cp


# feature scaling
def scale_features(x):
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    return x

def normalize(x):
    norm = Normalizer()
    x = norm.fit_transform(x)
    return x


def plot_pca(variance):
    plt.plot(variance)
    plt.xlabel('Number of components')
    plt.ylabel("Variance")
    plt.show()


def pca_selection(n_components, data):
    pca = PCA(n_components=n_components, random_state=200, whiten=True)
    data = pca.fit_transform(data)

    var_ratio = pca.explained_variance_ratio_*100
    print("Variance ratio - PCA\n", var_ratio)
    print("Total variance - PCA: ", np.sum(var_ratio))

    return data, var_ratio


def preprocess_data(data_movie):

    print(data_movie.info())

    check_missing_vals(data_movie)

    # impute data
    data_movie = impute(data_movie)
    check_missing_vals(data_movie)
    
    data_movie = data_movie.drop(columns=["Number of siblings", "Only child", "Village - town"])
  
    # one hot encoding
    data_movie = one_hot_encoding(data_movie, start_idx=13)
    data_cols = data_movie.columns
    data_movie_norm = normalize(data_movie)
    print(data_movie.head(3))

    return data_movie_norm, data_movie, data_cols
