import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

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
def one_hot_encoding(data):
    data_cp = data.copy()
    cols = data_cp.columns

    for i in range(14, data_cp.shape[1]):
        df = pd.get_dummies(data=data[cols[i]], prefix=cols[i], drop_first=True)
        data_cp = data_cp.drop(cols[i], axis=1)
        data_cp = pd.concat([data_cp, df], axis=1)

    return data_cp


# feature scaling
def scale_features(x):
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    return x


