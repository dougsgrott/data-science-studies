import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.animation as animation
import seaborn as sns

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

sns.set_style('whitegrid', {'legend.frameon':True, 'legend.facecolor': 'white'})
rc('animation', html='jshtml')

def plot_blobs(i):
    
    n_samples = [10, 100, 1000, 10000, 100000][i]
    centers = [2, 4, 6, 8, 10][i]
    cluster_std = [1, 1.5, 2, 3, 4, 5][i]
    center_box = [(-1.0, 1.0), (-4.0, 4.0), (-8.0, 8.0), (-15.0, 15.0), (-30.0, 30.0)][i]
    bbox = dict(boxstyle="square", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8))
    
    lim_list = [(-15, 15), (-15, 15), (-15, 15), center_box]
    title_list = [f"Plot 1 [n_samples = {n_samples}]", f"Plot 2 [centers = {centers}]", f"Plot 3 [cluster_std = {cluster_std}]", f"Plot 4 [center_box = {center_box}]"]
    for ax in axes:
        ax.clear()
        
    data1, labels1, centers1 = datasets.make_blobs(n_samples=n_samples, n_features=2, centers=3, cluster_std=1.0, center_box=(-10.0, 10.0), shuffle=True, random_state=42, return_centers=True)
    df1 = pd.DataFrame(np.hstack([data1, labels1.reshape(-1,1)]), columns=[f"f{i}" for i in range(data1.shape[1])] + ['class'])
    centers_df1 = pd.DataFrame(np.hstack([centers1, np.arange(len(centers1)).reshape(-1,1)]), columns=[f"f{i}" for i in range(data1.shape[1])] + ['class'])
    
    data2, labels2, centers2 = datasets.make_blobs(n_samples=500, n_features=2, centers=centers, cluster_std=1.0, center_box=(-10.0, 10.0), shuffle=True, random_state=42, return_centers=True)
    df2 = pd.DataFrame(np.hstack([data2, labels2.reshape(-1,1)]), columns=[f"f{i}" for i in range(data2.shape[1])] + ['class'])
    centers_df2 = pd.DataFrame(np.hstack([centers2, np.arange(len(centers2)).reshape(-1,1)]), columns=[f"f{i}" for i in range(data2.shape[1])] + ['class'])
    
    data3, labels3, centers3 = datasets.make_blobs(n_samples=500, n_features=2, centers=3, cluster_std=cluster_std, center_box=(-10.0, 10.0), shuffle=True, random_state=42, return_centers=True)
    df3 = pd.DataFrame(np.hstack([data3, labels3.reshape(-1,1)]), columns=[f"f{i}" for i in range(data3.shape[1])] + ['class'])
    centers_df3 = pd.DataFrame(np.hstack([centers3, np.arange(len(centers3)).reshape(-1,1)]), columns=[f"f{i}" for i in range(data3.shape[1])] + ['class'])
    
    data4, labels4, centers4 = datasets.make_blobs(n_samples=500, n_features=2, centers=3, cluster_std=1.0, center_box=center_box, shuffle=True, random_state=42, return_centers=True)
    df4 = pd.DataFrame(np.hstack([data4, labels4.reshape(-1,1)]), columns=[f"f{i}" for i in range(data4.shape[1])] + ['class'])
    centers_df4 = pd.DataFrame(np.hstack([centers4, np.arange(len(centers4)).reshape(-1,1)]), columns=[f"f{i}" for i in range(data4.shape[1])] + ['class'])
    
    sns.scatterplot(x='f0', y='f1', hue='class', data=df1, ax=axes[0], palette=sns.color_palette("hls", df1['class'].nunique()))
    sns.scatterplot(x='f0', y='f1', data=centers_df1, ax=axes[0], color='black', label="center")
    axes[0].text(x=0.5, y=0.02, s=f"n_samples={n_samples}, n_features=2, centers=3,\ncluster_std=1.0, center_box=(-10.0, 10.0)", ha="center", va="bottom", transform=axes[0].transAxes, bbox=bbox, fontsize=13)
    
    sns.scatterplot(x='f0', y='f1', hue='class', data=df2, ax=axes[1], palette=sns.color_palette("hls", df2['class'].nunique()))
    sns.scatterplot(x='f0', y='f1', data=centers_df2, ax=axes[1], color='black', label="center")
    axes[1].text(x=0.5, y=0.02, s=f"n_samples=500, n_features=2, centers={centers},\ncluster_std=1.0, center_box=(-10.0, 10.0)", ha="center", va="bottom", transform=axes[1].transAxes, bbox=bbox, fontsize=13)
    
    sns.scatterplot(x='f0', y='f1', hue='class', data=df3, ax=axes[2], palette=sns.color_palette("hls", df3['class'].nunique()))
    sns.scatterplot(x='f0', y='f1', data=centers_df3, ax=axes[2], color='black', label="center")
    axes[2].text(x=0.5, y=0.02, s=f"n_samples=500, n_features=2, centers=3,\ncluster_std={cluster_std}, center_box=(-10.0, 10.0)", ha="center", va="bottom", transform=axes[2].transAxes, bbox=bbox, fontsize=13)
    
    sns.scatterplot(x='f0', y='f1', hue='class', data=df4, ax=axes[3], palette=sns.color_palette("hls", df4['class'].nunique()))
    sns.scatterplot(x='f0', y='f1', data=centers_df4, ax=axes[3], color='black', label="center")
    axes[3].text(x=0.5, y=0.02, s=f"n_samples=500, n_features=2, centers=3,\ncluster_std=1.0, center_box={center_box}", ha="center", va="bottom", transform=axes[3].transAxes, bbox=bbox, fontsize=13)
    
    for ax, lim, title in zip(axes, lim_list, title_list):
        ax.set_xlim(lim[0], lim[1])
        ax.set_ylim(lim[0], lim[1])
        ax.legend(loc='upper right')
        ax.set_title(title, fontsize=18)
        ax.set_xlabel('$f_1$', fontsize=22)
        ax.set_ylabel('$f_2$', fontsize=22)
    axes[3].tick_params(axis="both", colors="red", labelsize=14)
    

def plot_moons(i):
    
    n_samples1 = [10, 100, 1000, 10000, 100000][i]
    n_samples2 = [(500, 500), (600, 400), (700, 300), (800, 200), (900, 100)][i]
    noise = [0, 0.03, 0.06, 0.09, 0.12][i]
    
    title_list = [f"Plot 1 [n_samples = {n_samples1}]", f"Plot 2 [n_samples = {n_samples2}]", f"Plot 3 [noise = {noise}]"]
    for ax, title in zip(axes, title_list):
        ax.clear()
        ax.legend(loc='upper right')
        ax.set_title(title, fontsize=18)
        ax.set_xlabel('$f_1$', fontsize=18)
        ax.set_ylabel('$f_2$', fontsize=18)
    bbox = dict(boxstyle="square", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8))
        
    data1, labels1 = datasets.make_moons(n_samples=n_samples1, shuffle=True, noise=0.04, random_state=42)
    df1 = pd.DataFrame(np.hstack([data1, labels1.reshape(-1,1)]), columns=["f0", "f1", "class"])
    
    data2, labels2 = datasets.make_moons(n_samples=n_samples2, shuffle=True, noise=0.04, random_state=42)
    df2 = pd.DataFrame(np.hstack([data2, labels2.reshape(-1,1)]), columns=["f0", "f1", "class"])
    
    data3, labels3 = datasets.make_moons(n_samples=1000, shuffle=True, noise=noise, random_state=42)
    df3 = pd.DataFrame(np.hstack([data3, labels3.reshape(-1,1)]), columns=["f0", "f1", "class"])
    
    sns.scatterplot(x='f0', y='f1', hue='class', data=df1, ax=axes[0], palette=sns.color_palette("hls", df1['class'].nunique()))
    sns.scatterplot(x='f0', y='f1', hue='class', data=df2, ax=axes[1], palette=sns.color_palette("hls", df2['class'].nunique()))    
    sns.scatterplot(x='f0', y='f1', hue='class', data=df3, ax=axes[2], palette=sns.color_palette("hls", df3['class'].nunique()))
    
    axes[0].text(x=0.5, y=0.02, s=f"n_samples={n_samples1}, noise=0.04", ha="center", va="bottom", transform=axes[0].transAxes, bbox=bbox, fontsize=13)
    axes[1].text(x=0.5, y=0.02, s=f"n_samples={n_samples2}, noise=0.04", ha="center", va="bottom", transform=axes[1].transAxes, bbox=bbox, fontsize=13)
    axes[2].text(x=0.5, y=0.02, s=f"n_samples=1000, noise={noise}", ha="center", va="bottom", transform=axes[2].transAxes, bbox=bbox, fontsize=13)


def plot_circles(i):
    
    n_samples = [10, 100, 1000, 10000, 100000][i]
    noise = [0, 0.03, 0.06, 0.09, 0.12][i]
    factor = [0, 0.2, 0.5, 0.7, 0.9][i]
    
    title_list = [f"Plot 1 [n_samples = {n_samples}]", f"Plot 2 [noise = {noise}]", f"Plot 3 [factor = {factor}]"]
    for ax, title in zip(axes, title_list):
        ax.clear()
        ax.legend(loc='upper right')
        ax.set_title(title, fontsize=18)
        ax.set_xlabel('$f_1$', fontsize=18)
        ax.set_ylabel('$f_2$', fontsize=18)
    bbox = dict(boxstyle="square", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8))
        
    data1, labels1 = datasets.make_circles(n_samples=n_samples, noise=0.03, random_state=42, factor=0.6)
    df1 = pd.DataFrame(np.hstack([data1, labels1.reshape(-1,1)]), columns=["f0", "f1", "class"])
    
    data2, labels2 = datasets.make_circles(n_samples=500, noise=noise, random_state=42, factor=0.6)
    df2 = pd.DataFrame(np.hstack([data2, labels2.reshape(-1,1)]), columns=["f0", "f1", "class"])
    
    data3, labels3 = datasets.make_circles(n_samples=500, noise=0.03, random_state=42, factor=factor)
    df3 = pd.DataFrame(np.hstack([data3, labels3.reshape(-1,1)]), columns=["f0", "f1", "class"])
    
    sns.scatterplot(x='f0', y='f1', hue='class', data=df1, ax=axes[0], palette=sns.color_palette("hls", df1['class'].nunique()))
    sns.scatterplot(x='f0', y='f1', hue='class', data=df2, ax=axes[1], palette=sns.color_palette("hls", df2['class'].nunique()))    
    sns.scatterplot(x='f0', y='f1', hue='class', data=df3, ax=axes[2], palette=sns.color_palette("hls", df3['class'].nunique()))
    
    axes[0].text(x=0.5, y=0.02, s=f"n_samples={n_samples}, noise=0.03, factor=0.6", ha="center", va="bottom", transform=axes[0].transAxes, bbox=bbox, fontsize=13)
    axes[1].text(x=0.5, y=0.02, s=f"n_samples=500, noise={noise}, factor=0.6", ha="center", va="bottom", transform=axes[1].transAxes, bbox=bbox, fontsize=13)
    axes[2].text(x=0.5, y=0.02, s=f"n_samples=500, noise=0.03, factor={factor}", ha="center", va="bottom", transform=axes[2].transAxes, bbox=bbox, fontsize=13)
    

def plot_classifications(i):
    
    n_informative = [1, 2, 3, 4, 5, 6, 7, 8][i]
    n_repeated = [0, 1, 2, 3, 4, 5, 6, 7][i]
    n_redundant = [0, 1, 2, 3, 4, 5, 6, 7][i]
    n_classes = [1, 2, 3, 4, 5, 6, 7, 8][i]
    n_clusters_per_class = [1, 2, 3, 4, 5, 6, 7, 8][i]
        
    title_list = [f"Plot 1 [n_informative={n_informative}]", f"Plot 2 [n_repeated={n_repeated}]", f"Plot 3 [n_redundant={n_redundant}]", f"Plot 4 [n_classes={n_classes}]", f"Plot 5 [n_clusters_per_class={n_clusters_per_class}]"]
    for ax in axes.flatten():
        ax.clear()
        
    for col in range(5):
        axes[0][col].set_title(title_list[col], fontsize=18)
    
    data1, labels1 = datasets.make_classification(n_samples=1000, n_features=8, n_informative=n_informative, n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=1, class_sep=50.0, random_state=42)
    df1 = pd.DataFrame(np.hstack([data1, labels1.reshape(-1,1)]), columns=[f"f{j}" for j in range(data1.shape[1])] + ['class'])
    
    data2, labels2 = datasets.make_classification(n_samples=1000, n_features=8, n_informative=1, n_redundant=0, n_repeated=n_repeated, n_classes=2, n_clusters_per_class=1, class_sep=50.0, random_state=42)
    df2 = pd.DataFrame(np.hstack([data2, labels2.reshape(-1,1)]), columns=[f"f{j}" for j in range(data2.shape[1])] + ['class'])
    
    data3, labels3 = datasets.make_classification(n_samples=1000, n_features=8, n_informative=1, n_redundant=n_redundant, n_repeated=0, n_classes=2, n_clusters_per_class=1, class_sep=50.0, random_state=42)
    df3 = pd.DataFrame(np.hstack([data3, labels3.reshape(-1,1)]), columns=[f"f{j}" for j in range(data3.shape[1])] + ['class'])
    
    data4, labels4 = datasets.make_classification(n_samples=1000, n_features=8, n_informative=n_informative, n_redundant=0, n_repeated=0, n_classes=n_classes, n_clusters_per_class=1, class_sep=50.0, random_state=42)
    df4 = pd.DataFrame(np.hstack([data4, labels4.reshape(-1,1)]), columns=[f"f{j}" for j in range(data4.shape[1])] + ['class'])
    
    data5, labels5 = datasets.make_classification(n_samples=1000, n_features=8, n_informative=n_informative, n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=n_clusters_per_class, class_sep=50.0, random_state=42)
    df5 = pd.DataFrame(np.hstack([data5, labels5.reshape(-1,1)]), columns=[f"f{j}" for j in range(data5.shape[1])] + ['class'])

    pd.plotting.radviz(df1, 'class', color=sns.color_palette("hls", df1['class'].nunique()), ax=axes[0][0])
    pd.plotting.parallel_coordinates(df1, 'class', color=sns.color_palette("hls", df1['class'].nunique()), ax=axes[1][0])

    pd.plotting.radviz(df2, 'class', color=sns.color_palette("hls", df2['class'].nunique()), ax=axes[0][1])
    pd.plotting.parallel_coordinates(df2, 'class', color=sns.color_palette("hls", df2['class'].nunique()), ax=axes[1][1])
    
    pd.plotting.radviz(df3, 'class', color=sns.color_palette("hls", df3['class'].nunique()), ax=axes[0][2])
    pd.plotting.parallel_coordinates(df3, 'class', color=sns.color_palette("hls", df3['class'].nunique()), ax=axes[1][2])
    
    pd.plotting.radviz(df4, 'class', color=sns.color_palette("hls", df4['class'].nunique()), ax=axes[0][3])
    pd.plotting.parallel_coordinates(df4, 'class', color=sns.color_palette("hls", df4['class'].nunique()), ax=axes[1][3])
    
    pd.plotting.radviz(df5, 'class', color=sns.color_palette("hls", df5['class'].nunique()), ax=axes[0][4])
    pd.plotting.parallel_coordinates(df5, 'class', color=sns.color_palette("hls", df5['class'].nunique()), ax=axes[1][4])


def plot_gaussian_quantiles(i):
    
    for ax in axes.flatten():
        ax.clear()
    
    n_samples = [10, 100, 1000, 10000][i%4]
    n_classes = [1, 2, 3, 4][i//4]
    
    data, labels = datasets.make_gaussian_quantiles(mean=np.array([1, 40, 800]), cov=1.0, n_samples=n_samples, n_features=3, n_classes=n_classes, shuffle=True, random_state=42)
    df = pd.DataFrame(np.hstack([data, labels.reshape(-1,1)]), columns=[f"f{i}" for i in range(data.shape[1])] + ['class'])

    sns.histplot(df, x='f0', hue='class', palette=sns.color_palette("hls", df['class'].nunique()), ax=axes[0][0], legend=False)
    sns.histplot(df, x='f1', hue='class', palette=sns.color_palette("hls", df['class'].nunique()), ax=axes[1][1])
    sns.histplot(df, x='f2', hue='class', palette=sns.color_palette("hls", df['class'].nunique()), ax=axes[2][2], legend=False)

    sns.scatterplot(data=df, x='f0', y='f1', hue='class', palette=sns.color_palette("hls", df['class'].nunique()), ax=axes[1][0], legend=False)
    sns.scatterplot(data=df, x='f0', y='f2', hue='class', palette=sns.color_palette("hls", df['class'].nunique()), ax=axes[2][0], legend=False)
    sns.scatterplot(data=df, x='f1', y='f2', hue='class', palette=sns.color_palette("hls", df['class'].nunique()), ax=axes[2][1], legend=False)

    for row, col in zip([0, 1], [0, 1]):
        axes[row][col].set_ylabel('')
        axes[row][col].set_xlabel('')
    axes[2][2].set_ylabel('')
    axes[1][0].set_xlabel('')

    for row, col in zip([0, 0, 1], [1, 2, 2]):
        axes[row][col].set_visible(False)


def plot_hastie_comparison(save: bool=False):
    X1, y1 = datasets.make_hastie_10_2(n_samples=1000, random_state=42)
    df1 = pd.DataFrame(np.hstack([X1, y1.reshape(-1,1)]), columns=[f"f{i}" for i in range(X1.shape[1])] + ['class'])
    X2, y2 = datasets.make_gaussian_quantiles(n_samples=1000, n_features=10, n_classes=2, random_state=42)
    df2 = pd.DataFrame(np.hstack([X2, y2.reshape(-1,1)]), columns=[f"f{i}" for i in range(X2.shape[1])] + ['class'])
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 3))
    
    sns.histplot(x='f0', hue='class', data=df1, ax=axes[0], palette=sns.color_palette("hls", df1['class'].nunique()))
    sns.scatterplot(x='f0', y='f1', hue='class', data=df1, ax=axes[1], palette=sns.color_palette("hls", df1['class'].nunique()))
    sns.histplot(x='f0', hue='class', data=df2, ax=axes[2], palette=sns.color_palette("hls", df2['class'].nunique()))
    sns.scatterplot(x='f0', y='f1', hue='class', data=df2, ax=axes[3], palette=sns.color_palette("hls", df2['class'].nunique()))

    axes[0].set_title("Histogram for Hastie 10_2", fontsize=15)
    axes[1].set_title("Scatterplot for Hastie 10_2", fontsize=15)
    axes[2].set_title("Histogram for Quantile", fontsize=15)
    axes[3].set_title("Scatterplot for Quantile", fontsize=15)

    if save:
        os.makedirs(os.path.dirname("./generating_datasets_using_scikit_learn/generated_assets/"), exist_ok=True)
        plt.savefig("./generating_datasets_using_scikit_learn/generated_assets/make_hastie_10_2 comparison.png", format="png")
    plt.close()


def plot_multilabel_classification(i):
    n_classes = i+1
    X, Y = datasets.make_multilabel_classification(n_samples=300, n_features=20, n_classes=3, n_labels=2, length=50, allow_unlabeled=True, random_state=42)
    df = pd.DataFrame(np.hstack([X, Y]), columns=[f"f{i}" for i in range(X.shape[1])] + [f"class {i}" for i in range(Y.shape[1])])

    data_embedded_df = pd.DataFrame(PCA(n_components=2).fit_transform(X))
    data_embedded_df.columns = ['F0', 'F1']
    
    for i in range(Y.shape[1]):
        data_embedded_df[f"class {i}"] = df[f"class {i}"]
        
    val_f1_min, val_f1_max = np.min(data_embedded_df['F1'])-1, np.max(data_embedded_df['F1'])+1
    val_f0_min, val_f0_max = np.min(data_embedded_df['F0'])-1, np.max(data_embedded_df['F0'])+1

    palette = sns.color_palette("hls", 2**3)
    unique_colors = np.array(palette.as_hex())
    df['color'] = unique_colors.take((Y[:, :3] * np.array([2**n for n in range(3)])).sum(axis=1))
    df['classes'] = df[['class 0', 'class 1', 'class 2']].apply(lambda row: str(row.values), axis=1)
    df['classes_order'] = (Y[:, :3] * np.array([2**n for n in range(3)])).sum(axis=1)

    classes_order = ['[0. 0. 0.]', '[1. 0. 0.]', '[0. 1. 0.]', '[1. 1. 0.]', '[0. 0. 1.]', '[1. 0. 1.]', '[0. 1. 1.]', '[1. 1. 1.]']

    for col in ['class 0', 'class 1', 'class 2', 'color', 'classes', 'classes_order']:
        data_embedded_df[col] = df[col]

    title_list = ["Plot 1 - Multilabel represented through color, size and markers", "Plot 2 - Multilabel represented using only colors"]
    for ax in axes:
        ax.clear()
        
    if n_classes == 1:
        data_embedded_df = data_embedded_df[data_embedded_df['class 1'] == 0]
        data_embedded_df = data_embedded_df[data_embedded_df['class 2'] == 0]
        palette = palette[:2]
        classes_order = classes_order[:2]
        sns.scatterplot(data=data_embedded_df, x='F0', y='F1', hue='class 0', s=200, ax=axes[0])
    if n_classes == 2:
        data_embedded_df = data_embedded_df[data_embedded_df['class 2'] == 0]
        palette = palette[:4]
        classes_order = classes_order[:4]
        sns.scatterplot(data=data_embedded_df, x='F0', y='F1', hue='class 0', style="class 1", s=200, ax=axes[0])
    if n_classes == 3:
        sns.scatterplot(data=data_embedded_df, x='F0', y='F1', hue='class 0', style="class 1", size='class 2', sizes=(50, 200), ax=axes[0])

    sns.scatterplot(data=data_embedded_df, x='F0', y='F1', hue='classes', hue_order=classes_order, palette=palette, s=200, ax=axes[1])

    for ax, title in zip(axes, title_list):
        ax.set_xlim(val_f0_min, val_f0_max)
        ax.set_ylim(val_f1_min, val_f1_max)
        ax.legend(loc='upper right')
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('$F_1$', fontsize=18)
        ax.set_ylabel('$F_2$', fontsize=18)
    

def model_and_plot_linear_regression(save: bool=False):
    generators = [datasets.make_regression, datasets.make_sparse_uncorrelated]
    titles = ["make_regression", "make_sparse_uncorrelated"]

    fig, axes = plt.subplots(nrows=1, ncols=len(generators)*2, figsize=(15,5))

    for i, generator in enumerate(generators):
        X, y = generator(n_samples=1000)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

        lin_regr = LinearRegression()
        lin_regr.fit(X_train, y_train)
        lin_r2_train, lin_r2_test = lin_regr.score(X_train, y_train), lin_regr.score(X_test, y_test)
        lin_y_hat = lin_regr.predict(X_test)
        
        nlin_regr = make_pipeline(StandardScaler(), MLPRegressor(random_state=1, max_iter=10000))
        nlin_regr.fit(X_train, y_train)
        nlin_r2_train, nlin_r2_test = nlin_regr.score(X_train, y_train), nlin_regr.score(X_test, y_test)
        nlin_y_hat = nlin_regr.predict(X_test)

        sns.scatterplot(x=y_test, y=lin_y_hat,  ax=axes[2*i], s=80)
        sns.scatterplot(x=y_test, y=nlin_y_hat, ax=axes[2*i+1], s=80, color='green')

        bbox = dict(boxstyle="square", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8))
        axes[2*i].text(x=1.0, y=0.02, s=f"$R^2$ (train) = {lin_r2_train:.2f}\n$R^2$ (test) = {lin_r2_test:.2f}", ha="right", va="bottom", transform=axes[2*i].transAxes, bbox=bbox, fontsize=18)
        axes[2*i+1].text(x=1.0, y=0.02, s=f"$R^2$ (train) = {nlin_r2_train:.2f}\n$R^2$ (test) = {nlin_r2_test:.2f}", ha="right", va="bottom", transform=axes[2*i+1].transAxes, bbox=bbox, fontsize=18)
        axes[2*i].set_title(f"{titles[i]}\nmodeled by Linear Regression", fontsize=14)
        axes[2*i+1].set_title(f"{titles[i]}\nmodeled by Neural Network", fontsize=14)
        axes[0].set_ylabel("$y_{prediction}$", fontsize=20)
        axes[2*i].set_xlabel("$y_{target}$", fontsize=20)
        axes[2*i+1].set_xlabel("$y_{target}$", fontsize=20)

    if save:
        os.makedirs(os.path.dirname("./generating_datasets_using_scikit_learn/generated_assets/"), exist_ok=True)
        plt.savefig("./generating_datasets_using_scikit_learn/generated_assets/make_regression and make_sparse_uncorrelated.png", format="png")
    plt.close()


def model_and_plot_nonlinear_regression(save: bool=False):
    generators = [datasets.make_friedman1, datasets.make_friedman2, datasets.make_friedman3]
    titles = ["make_friedman1", "make_friedman2", "make_friedman3"]

    fig, axes = plt.subplots(nrows=1, ncols=len(generators)*2, figsize=(25,5))

    for i, generator in enumerate(generators):
        X, y = generator(n_samples=1000)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

        lin_regr = LinearRegression()
        lin_regr.fit(X_train, y_train)
        lin_r2_train, lin_r2_test = lin_regr.score(X_train, y_train), lin_regr.score(X_test, y_test)
        lin_y_hat = lin_regr.predict(X_test)
        
        nlin_regr = make_pipeline(StandardScaler(), MLPRegressor(random_state=1, max_iter=10000))
        nlin_regr.fit(X_train, y_train)
        nlin_r2_train, nlin_r2_test = nlin_regr.score(X_train, y_train), nlin_regr.score(X_test, y_test)
        nlin_y_hat = nlin_regr.predict(X_test)

        sns.scatterplot(x=y_test, y=lin_y_hat,  ax=axes[2*i], s=80)
        sns.scatterplot(x=y_test, y=nlin_y_hat, ax=axes[2*i+1], s=80, color='green')

        bbox = dict(boxstyle="square", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8))
        axes[2*i].text(x=1.0, y=0.02, s=f"$R^2$ (train) = {lin_r2_train:.2f}\n$R^2$ (test) = {lin_r2_test:.2f}", ha="right", va="bottom", transform=axes[2*i].transAxes, bbox=bbox, fontsize=18)
        axes[2*i+1].text(x=1.0, y=0.02, s=f"$R^2$ (train) = {nlin_r2_train:.2f}\n$R^2$ (test) = {nlin_r2_test:.2f}", ha="right", va="bottom", transform=axes[2*i+1].transAxes, bbox=bbox, fontsize=18)
        axes[2*i].set_title(f"{titles[i]}\nmodeled by Linear Regression", fontsize=14)
        axes[2*i+1].set_title(f"{titles[i]}\nmodeled by Neural Networks", fontsize=14)
        axes[0].set_ylabel("$y_{prediction}$", fontsize=20)
        axes[2*i].set_xlabel("$y_{target}$", fontsize=20)
        axes[2*i+1].set_xlabel("$y_{target}$", fontsize=20)

    if save:
        os.makedirs(os.path.dirname("./generating_datasets_using_scikit_learn/generated_assets/"), exist_ok=True)
        plt.savefig("./generating_datasets_using_scikit_learn/generated_assets/make_friedmans.png", format="png")
    plt.close()


def plot_s_curve_swiss_roll(i):

    angle = np.linspace(start=0, stop=360, num=total_frames, endpoint=False)[i]
    noise1 = np.linspace(start=0, stop=0.5, num=total_frames, endpoint=False)[i]
    noise2 = np.linspace(start=0, stop=2, num=total_frames, endpoint=False)[i]
    
    title_list = ["Plot 1A - S Curve", "Plot 2A - Swiss Roll", f"Plot 1B [noise={noise1:.2f}]", f"Plot 2B [noise={noise2:.2f}]"]
    for ax, title in zip([ax1, ax2, ax3, ax4], title_list):
        ax.clear()
        ax.set_title(title)

    X1, t1 = datasets.make_s_curve(n_samples=5000, noise=0.0, random_state=42)
    df1 = pd.DataFrame(np.hstack([X1, t1.reshape(-1,1)]), columns=[f"f{i}" for i in range(X1.shape[1])] + ['class'])

    X2, t2 = datasets.make_swiss_roll(n_samples=5000, noise=0.0, random_state=42)
    df2 = pd.DataFrame(np.hstack([X2, t2.reshape(-1,1)]), columns=[f"f{i}" for i in range(X2.shape[1])] + ['class'])
    
    X3, t3 = datasets.make_s_curve(n_samples=5000, noise=noise1, random_state=42)
    df3 = pd.DataFrame(np.hstack([X3, t3.reshape(-1,1)]), columns=[f"f{i}" for i in range(X3.shape[1])] + ['class'])

    X4, t4 = datasets.make_swiss_roll(n_samples=5000, noise=noise2, random_state=42)
    df4 = pd.DataFrame(np.hstack([X4, t4.reshape(-1,1)]), columns=[f"f{i}" for i in range(X4.shape[1])] + ['class'])

    ax1.scatter3D(df1['f0'], df1['f1'], df1['f2'], c=t1, cmap=plt.cm.Spectral)
    ax2.scatter3D(df2['f0'], df2['f1'], df2['f2'], c=t2, cmap=plt.cm.Spectral)
    ax3.scatter3D(df3['f0'], df3['f1'], df3['f2'], c=t3, cmap=plt.cm.Spectral)
    ax4.scatter3D(df4['f0'], df4['f1'], df4['f2'], c=t4, cmap=plt.cm.Spectral)

    ax1.view_init(30,angle)
    ax2.view_init(30,angle)


def plot_matrices(save: bool=False):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    my_cmap = sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True)

    lrm_df = pd.DataFrame(datasets.make_low_rank_matrix(n_samples=50, n_features=50, effective_rank=10, tail_strength=0.5, random_state=42))
    sns.heatmap(lrm_df, xticklabels=lrm_df.columns, annot=False, linewidths=0.0, ax=axes[0][0], cbar=True, cmap=my_cmap)
    axes[1][0].spy(lrm_df, precision=0.0)
    axes[1][0].grid(False)
    axes[0][0].set_title("Low Rank Matrix", fontsize=18)

    spd_df = pd.DataFrame(datasets.make_spd_matrix(n_dim=50, random_state=42))
    sns.heatmap(spd_df, xticklabels=spd_df.columns, annot=False, linewidths=0.0, ax=axes[0][1], cbar=True, cmap=my_cmap)
    axes[1][1].spy(spd_df, precision=0.0)
    axes[1][1].grid(False)
    axes[0][1].set_title("SPD Matrix", fontsize=18)

    sspd_df = pd.DataFrame(datasets.make_sparse_spd_matrix(dim=50, alpha=0.95, norm_diag=False, smallest_coef=0.0, largest_coef=1.0, random_state=42))
    sns.heatmap(sspd_df, xticklabels=sspd_df.columns, annot=False, linewidths=0.0, ax=axes[0][2], cbar=True, cmap=my_cmap)
    axes[1][2].spy(sspd_df, precision=0.0)
    axes[1][2].grid(False)
    axes[0][2].set_title("Sparse SPD Matrix", fontsize=18)

    np.random.seed(3)
    rand_df = pd.DataFrame(np.random.rand(50,50))
    sns.heatmap(rand_df, xticklabels=rand_df.columns, annot=False, linewidths=0.0, ax=axes[0][3], cbar=True, cmap=my_cmap)
    axes[1][3].spy(rand_df, precision=0.0)
    axes[1][3].grid(False)
    axes[0][3].set_title("Random numbers", fontsize=18)

    bbox = dict(boxstyle="square", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8))
    for i in range(4):
        axes[1][i].text(x=0.5, y=0.05, s="White squares represent zeros", ha="center", va="bottom", transform=axes[1][i].transAxes, bbox=bbox, fontsize=13)

    if save:
        os.makedirs(os.path.dirname("./generating_datasets_using_scikit_learn/generated_assets/"), exist_ok=True)
        plt.savefig("./generating_datasets_using_scikit_learn/generated_assets/matrices.png", format="png")
    plt.close()


def plot_sparse_coded_signal(save: bool=False):
    data, dictionary, code = datasets.make_sparse_coded_signal(n_samples=1, n_components=512, n_features=150, n_nonzero_coefs=40, random_state=42)

    (idx,) = code.nonzero()

    fig, axes = plt.subplots(figsize=(10, 3))
    axes.set_title("Sparse signal")
    axes.stem(idx, code[idx], use_line_collection=True)

    axes.set_title("Sparse Signal", fontsize=16)
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)

    if save:
        os.makedirs(os.path.dirname("./generating_datasets_using_scikit_learn/generated_assets/"), exist_ok=True)
        plt.savefig("./generating_datasets_using_scikit_learn/generated_assets/make_sparse_coded_signal.png", format="png")
    plt.close()


if "__main__"==__name__:

    os.makedirs(os.path.dirname("./generating_datasets_using_scikit_learn/generated_assets/"), exist_ok=True)

    ###################################################
    #    Clustering
    print("Creating visualizations for Clustering")

    ## make_blobs
    total_frames = 5
    fig, axes = plt.subplots(1, 4, figsize=(25, 5))
    anim = animation.FuncAnimation(fig, plot_blobs, frames=total_frames, interval=1000, blit=False)
    anim.save('./generating_datasets_using_scikit_learn/generated_assets/make_blobs.gif', writer='imagemagick')
    plt.close()

    ## make_moons
    total_frames = 5
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    anim = animation.FuncAnimation(fig, plot_moons, frames=total_frames, interval=1000, blit=False)
    anim.save('./generating_datasets_using_scikit_learn/generated_assets/make_moons.gif', writer='imagemagick')
    plt.close()

    ## make_circles
    total_frames = 5
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    anim = animation.FuncAnimation(fig, plot_circles, frames=total_frames, interval=1000, blit=False)
    anim.save('./generating_datasets_using_scikit_learn/generated_assets/make_circles.gif', writer='imagemagick')
    plt.close()

    ###################################################
    #    Single-Label Classification
    print("Creating visualizations for Single-Label Classification")

    ## make_classification
    total_frames = 8
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(24, 6))
    anim = animation.FuncAnimation(fig, plot_classifications, frames=total_frames, interval=1000, blit=False)
    anim.save('./generating_datasets_using_scikit_learn/generated_assets/make_classification.gif', writer='imagemagick')
    plt.close()

    ## make_gaussian_quantile
    total_frames = 16
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
    anim = animation.FuncAnimation(fig, plot_gaussian_quantiles, frames=total_frames, interval=1000, blit=False)
    anim.save('./generating_datasets_using_scikit_learn/generated_assets/make_gaussian_quantile.gif', writer='imagemagick')
    plt.close()

    ## make_hastie_10_2
    plot_hastie_comparison(save=True)

    ###################################################
    #    Multi-Label Classification
    print("Creating visualizations for Multi-Label Classification")

    ## make_multilabel_classification
    total_frames = 3
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,4))
    anim = animation.FuncAnimation(fig, plot_multilabel_classification, frames=total_frames, interval=2000, blit=False)
    anim.save('./generating_datasets_using_scikit_learn/generated_assets/make_multilabel_classification.gif', writer='imagemagick')
    plt.close()

    ###################################################
    #     Linear Regression
    print("Creating visualizations for Linear Regression")

    ## make_regression and make_sparse_uncorrelated
    model_and_plot_linear_regression(save=True)

    ###################################################
    #    Non-Linear Regression
    print("Creating visualizations for Non-Linear Regression")

    ## make_friedman1, make_friedman2 and make_friedman3
    model_and_plot_nonlinear_regression(save=True)

    ###################################################
    #    Manifold Generation
    print("Creating visualizations for Manifold Generation")

    ## make_s_curve and make_swiss_roll
    total_frames = 12*2*2
    fig = plt.figure(figsize=(15,5))
    ax1 = fig.add_subplot(141, projection='3d')
    ax2 = fig.add_subplot(142, projection='3d')
    ax3 = fig.add_subplot(143, projection='3d')
    ax4 = fig.add_subplot(144, projection='3d')
    anim = animation.FuncAnimation(fig, plot_s_curve_swiss_roll, frames=total_frames, interval=100, blit=False)
    anim.save('./generating_datasets_using_scikit_learn/generated_assets/make_manifold.gif', writer='imagemagick')
    plt.close()

    ###################################################
    #    Decomposition
    print("Creating visualizations for Decomposition")

    ## make_low_rank_matrix, make_spd_matrix and make_sparse_spd_matrix
    plot_matrices(save=True)

    ## make_sparse_coded_signal
    plot_sparse_coded_signal(save=True)

    print("EOL")