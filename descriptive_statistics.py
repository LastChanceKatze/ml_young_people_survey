import preprocess as pp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_boxplot(data):
    data.plot(kind='box', figsize=(10, 10), subplots=True, layout=(2, 7),
              sharex=False, sharey=False,
              title='Box Plot for each input variable')
    plt.subplots_adjust(wspace=0.5)
    plt.savefig('./graphs/features_box_plot.jpg')
    plt.show()


def plot_countplots(data):
    num_rows = 4
    num_cols = 5
    fig, axs = plt.subplots(num_rows, num_cols)
    plt.subplots_adjust(hspace=0.5, wspace=.8, top=0.95)

    for i in range(0, num_rows):
        for j in range(0, num_cols):
            index = j * num_rows + i
            if index < data.shape[1]:
                col_name = data.columns[index]
                sns.countplot(x=col_name, data=data, ax=axs[i][j])

    plt.savefig("./graphs/countplots.jpg")
    plt.show()


def plot_barplot(data, col_x, col_y):
    plt.figure(figsize=(3, 3))
    sns.barplot(x=col_x, y=col_y, data=data)

    plt.savefig("./graphs/barplot_"+col_x+"_"+col_y+".png")
    plt.show()


def correlation_matrix(data):
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, ax=ax)
    plt.savefig("./graphs/correlation_matrix.png")
    plt.show()


def exploratory_analysis():
    data = pd.read_csv("dataset/responses.csv")
    # select movie columns
    movie_columns = ["Movies", "Horror", "Thriller", "Comedy", "Romantic",
                     "Sci-fi", "War", "Fantasy/Fairy tales", "Animated",
                     "Documentary", "Western", "Action", "Age",
                     "Number of siblings", "Gender", "Education",
                     "Only child", "Village - town", "House - block of flats"]

    data_movie = data[movie_columns]
    print("Describe data: \n" + str(data.describe()))
    print("----------------------------------------------------------")

    data_movie = pp.impute(data_movie)
    data_movie_encoded = pp.encode_labels(data_movie)

    """
    correlation matrix
    """
    # correlation_matrix(data_movie_encoded)

    """
    box plots for each variable
    outliers can be seen
    as well as data range
    """
    # plot_boxplot(data_movie)

    """
    count plot for each variable
    """
    # plot_countplots(data_movie)

    """
    specific count plots
    """
    plot_barplot(data_movie, col_x="Gender", col_y="War")
    plot_barplot(data_movie, col_x="Gender", col_y="Action")
    plot_barplot(data_movie, col_x="Gender", col_y="Romantic")
    plot_barplot(data_movie, col_x="House - block of flats", col_y="Horror")


# exploratory_analysis()
