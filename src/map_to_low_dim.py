#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: LL
# @File：map_to_low_dim.py
"""
Map the high-dimensional phase or element vector to low-dimensional vector
"""
import math
import os
from collections import Counter
import seaborn as sns
import numpy as np
import pandas as pd
import scipy
from matplotlib.colors import LinearSegmentedColormap, Normalize
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
import umap
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.cluster import OPTICS
from sklearn.cluster import BisectingKMeans
import scienceplots
import joblib

# Arial
plt.style.use(['science', 'no-latex'])
plt.rcParams['font.family'] = 'Arial'


def load_data(csv_file, source_mapping, mode=None):
    data = pd.read_csv(csv_file)
    source = data['Source'].map(label_to_index).tolist()
    index = data['Index'].tolist()
    data = data.drop(['Index', 'Source'], axis=1)

    return data, source, index


def apply_umap(data, n_components=2, n_neighbors=10, min_dist=0.2, spread=1.0):
    # Apply UMAP for dimensionality reduction
    reducer = umap.UMAP(n_components=n_components, random_state=42, metric='cosine', n_neighbors=n_neighbors, min_dist=min_dist, spread=spread)
    umap_result = reducer.fit_transform(data)
    if np.isnan(umap_result).any():
        print("Warning: UMAP result contains NaN values. Applying imputation.")
        from sklearn.impute import SimpleImputer
        imp = SimpleImputer(strategy='mean')
        umap_result = imp.fit_transform(umap_result)

    return umap_result, reducer


def apply_mds(data, n_components=2):
    # Apply MDS for dimensionality reduction
    mds = MDS(n_components=n_components, random_state=42)
    mds_result = mds.fit_transform(data)
    return mds_result, mds


def apply_pca(data, n_components=None):
    pca = PCA(n_components=n_components, random_state=42)
    transformed_data = pca.fit_transform(data)

    explained_variance_ratio = pca.explained_variance_ratio_
    components = pca.components_

    return transformed_data, pca, explained_variance_ratio, components

def apply_kmeans_clustering(data, num_clusters=3):
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data)
    # Generate cluster borders using meshgrid sampling and prediction
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

    point = np.vstack([xx.ravel(), yy.ravel()]).T.tolist()
    # Const 8-bit data type
    point = np.array(point, dtype=np.float64)
    Z = kmeans.predict(point)
    Z = np.array(Z).reshape(xx.shape)

    cluster_centers_ = kmeans.cluster_centers_
    inertia_ = kmeans.inertia_
    return cluster_labels, xx, yy, Z, cluster_centers_, inertia_


# Apply bi-kmeans clustering
def apply_bi_kmeans_clustering(data, num_clusters):
    bi_kmeans = BisectingKMeans(n_clusters=num_clusters, random_state=42, init='k-means++')
    cluster_labels = bi_kmeans.fit_predict(data)

    # Generate cluster borders using meshgrid sampling and prediction
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    point = np.vstack([xx.ravel(), yy.ravel()]).T

    # Convert to float32 to match sklearn's expected data type
    point = point.astype(np.float32)

    Z = bi_kmeans.predict(point)
    Z = Z.reshape(xx.shape)

    cluster_centers_ = bi_kmeans.cluster_centers_
    inertia_ = bi_kmeans.inertia_
    return cluster_labels, xx, yy, Z, cluster_centers_, inertia_


# Use Optics to cluster
def apply_optics_clustering(data, num_clusters=3):
    optics = OPTICS(min_samples=10, xi=0.05, min_cluster_size=0.05)
    cluster_labels = optics.fit_predict(data)
    # Generate cluster borders using meshgrid sampling and prediction
    x_min, x_max = data[:, 0].min() - 0.1, data[:, 0].max() + 0.1
    y_min, y_max = data[:, 1].min() - 0.1, data[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

    point = np.vstack([xx.ravel(), yy.ravel()]).T.tolist()
    # Const 8-bit data type
    point = np.array(point, dtype=np.float64)
    Z = optics.predict(point)
    Z = np.array(Z).reshape(xx.shape)

    cluster_centers_ = optics.cluster_centers_
    return cluster_labels, xx, yy, Z, cluster_centers_


def plot_results_pv(df, labels, x, y, Z, cluster_centers_, inertia_, method_name, output_path=None):
    # Plot the reduced data points with colored-coded clusters, cluster boundaries, and centers
    plt.figure(figsize=(7.2, 5.4))
    cmap_name = 'custom_cmap'
    colors = ['#ffaaa5', '#ffd3b6', '#dcedc1', '#a8e6cf']
    custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)
    plt.contourf(x, y, Z, cmap=custom_cmap, alpha=0.4)
    plt.contour(x, y, Z, colors="lightgrey", linewidths=1.5, linestyles='dashed', alpha=0.8)

    sources = sorted(df['Source'].unique())
    num_sources = len(sources)

    norm = Normalize(vmin=0, vmax=num_sources - 1)

    source_colors = plt.cm.rainbow(np.linspace(0, 1, num_sources))
    source_color_map = dict(zip(sources, source_colors))
    max_x, min_x = max(df['Dim_1']) + 0.25, min(df['Dim_1']) - 0.25
    max_y, min_y = max(df['Dim_2']) + 0.25, min(df['Dim_2']) - 0.25

    for source in sources:
        subset = df[df['Source'] == source]
        plt.scatter(subset['Dim_1'], subset['Dim_2'], color=source_color_map[source], label=source, edgecolors='white',
                    linewidths=0.5, s=100)
        for idx, row in subset.iterrows():
            plt.text(row['Dim_1'], row['Dim_2'],
                     str(int(source)),
                     fontsize=8,
                     ha='center',
                     va='center',
                     color='white',
                     weight='bold',
                     alpha=0.9)

    plt.scatter(cluster_centers_[:, 0], cluster_centers_[:, 1], c='red', marker='X', s=50, label='Centers')
    # Text of inertia
    plt.text(0.97, 0.03, 'SSE: {}'.format(round(inertia_, 1)), transform=plt.gca().transAxes, size=10,
             horizontalalignment='right')
    plt.xlabel(f'{method_name}_1', fontsize=15, labelpad=10)
    plt.ylabel(f'{method_name}_2', fontsize=15, labelpad=10)
    plt.legend(fontsize=8, markerscale=0.7, ncol=2, handletextpad=0.2, columnspacing=0.5, loc='lower right', bbox_to_anchor=(1, 0.05))
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    plt.tight_layout()
    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.5)
    ax = plt.gca()

    ax.tick_params(axis='both', which='both', top=False, right=False)
    ax.tick_params(axis='both', which='major', length=5, width=1.5, direction='out')
    ax.tick_params(axis='both', which='minor', length=3, width=0.5, direction='out')

    plt.savefig(output_path, dpi=450, bbox_inches='tight')


def plot_results_pvt(df, labels, x, y, Z, cluster_centers_, inertia_, method_name, output_path=None):
    # Plot the reduced data points with colored-coded clusters, cluster boundaries, and centers
    plt.figure(figsize=(7.2, 5.4))
    cmap_name = 'custom_cmap'
    colors = ['#ffaaa5', '#ffd3b6', '#dcedc1', '#a8e6cf']
    custom_cmap_1 = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)
    plt.contour(x, y, Z, colors="lightgrey", linewidths=1.5, linestyles='dashed', alpha=0.8)
    plt.contourf(x, y, Z, cmap=custom_cmap_1, alpha=0.4)
    colors = ['#FA8072', '#FF8C00', '#a4d247', '#32CD32']
    custom_cmap_2 = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)

    plt.scatter(df['Dim_1'], df['Dim_2'], c=labels, cmap=custom_cmap_2, s=50, edgecolors='white', linewidths=0.5,
                alpha=0.8)
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    cluster_sizes = df['Cluster'].value_counts().to_dict()

    top_phases_per_cluster = {}
    for cluster in np.unique(labels):
        cluster_df = df[df['Cluster'] == cluster]
        phase_counts = Counter(cluster_df['Phase'])

        top_phases_per_cluster = {}
        for cluster in np.unique(labels):
            cluster_df = df[df['Cluster'] == cluster]
            phase_counts = Counter(cluster_df['Phase'])
            n_annotations = 10 if cluster_sizes[cluster] > np.median(list(cluster_sizes.values())) else 5
            top_phases = [phase for phase, count in phase_counts.most_common(n_annotations)]
            top_phases_per_cluster[cluster] = top_phases

            for i, phase in enumerate(top_phases):
                phase_points = cluster_df[cluster_df['Phase'] == phase]

                if not phase_points.empty:
                    distances = np.sqrt((phase_points['Dim_1'] - cluster_centers_[cluster, 0]) ** 2 +
                                        (phase_points['Dim_2'] - cluster_centers_[cluster, 1]) ** 2)
                    closest_idx = distances.idxmin()
                    point_x = phase_points.loc[closest_idx, 'Dim_1']
                    point_y = phase_points.loc[closest_idx, 'Dim_2']

                    angle = 2 * math.pi * i / len(top_phases)
                    for distance in [3]:
                        text_x = point_x + distance * math.cos(angle)
                        text_y = point_y + distance * math.sin(angle)

                        if (xlim[0] <= text_x <= xlim[1]) and (ylim[0] <= text_y <= ylim[1]):
                            break
                    else:
                        text_x = np.clip(point_x + 15 * math.cos(angle), xlim[0], xlim[1])
                        text_y = np.clip(point_y + 15 * math.sin(angle), ylim[0], ylim[1])

                    ha = 'left' if math.cos(angle) >= 0 else 'right'
                    va = 'bottom' if math.sin(angle) >= 0 else 'top'

                    plt.annotate(f"{phase}",
                                 xy=(point_x, point_y),
                                 xytext=(text_x, text_y),
                                 textcoords='data',
                                 bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.2, ec='gray'),
                                 arrowprops=dict(arrowstyle='->', color='gray', linewidth=0.8),
                                 fontsize=9,
                                 ha=ha,
                                 va=va)

    # Text of inertia
    plt.text(0.97, 0.03, 'SSE: {}'.format(round(inertia_, 1)), transform=plt.gca().transAxes, size=10,
             horizontalalignment='right')
    plt.xlabel(f'{method_name}_1', fontsize=15, labelpad=10)
    plt.ylabel(f'{method_name}_2', fontsize=15, labelpad=10)

    plt.tight_layout()
    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.5)
    ax = plt.gca()

    ax.tick_params(axis='both', which='both', top=False, right=False)

    ax.tick_params(axis='both', which='major', length=5, width=1.5, direction='out')
    ax.tick_params(axis='both', which='minor', length=3, width=0.5, direction='out')



    plt.savefig(output_path, dpi=450, bbox_inches='tight')


def draw_elbow(data, max_k=10, output_path=None):
    """
    Draw elbow curve for clustering
    """
    inertia = []
    k_values = range(1, max_k + 1)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(8, 6))
    plt.plot(k_values, inertia, marker='o')
    # plt.title('Elbow Method for Optimal k', fontsize=20)
    plt.xlabel('Number of clusters (k)', fontsize=15)
    plt.ylabel('SSE', fontsize=15)
    plt.xticks(k_values)
    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.5)
    ax = plt.gca()

    ax.tick_params(axis='both', which='both', top=False, right=False)
    ax.tick_params(axis='both', which='major', length=5, width=1.5, direction='out')
    ax.tick_params(axis='both', which='minor', length=3, width=0.5, direction='out')

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    # Specify the path to your CSV file
    csv_file = "../data/phase_vector.csv"
    os.makedirs("../output/dim_reduction", exist_ok=True)
    label_encoder_path = "../output/hpo_exp/label_encoder.pkl"

    label_encoder = joblib.load(label_encoder_path)
    label_to_index = {label: i for i, label in enumerate(label_encoder.classes_)}


    # Low dimensional mapping of solid wastes
    n_components = 2  # Adjust this parameter
    n_neighbors = 10 # Adjust this parameter
    min_dist = 0.2  # Adjust this parameter
    spread = 1  # Adjust this parameter
    output_path = "../output/dim_reduction/phase_vector_umap.csv"
    data, source, index = load_data(csv_file, label_to_index)
    # Apply UMAP
    map_result, umap_reducer = apply_umap(data, n_components, n_neighbors, min_dist, spread)

    # Draw elbow curve
    draw_elbow(map_result, max_k=10, output_path='../output/dim_reduction/elbow_curve_umap.png')

    num_clusters = 3
    map_labels, x_umap, y_umap, Z_umap, umap_cluster_centers_, inertia_ = apply_bi_kmeans_clustering(map_result,
                                                                                                      num_clusters)
    # Save index and cluster labels as csv file
    df = pd.DataFrame({'Index': index,
                       'Dim_1': map_result[:, 0],
                       'Dim_2': map_result[:, 1],
                       'Cluster': map_labels,
                       'Source': source
                       })
    # Concatenate the data
    df = pd.concat([df, data], axis=1)
    df.to_csv("../output/dim_reduction/phase_vector_umap.csv", index=False, encoding='utf-8')
    plot_results_pv(df, map_labels, x_umap, y_umap, Z_umap, umap_cluster_centers_, inertia_, 'UMAP',
                 output_path='../output/dim_reduction/phase_vector_umap_clustering.png')


    # Low dimensional mapping of mineral phases
    csv_file = "../data/phase_vector_transposed.csv"
    df = pd.read_csv(csv_file)
    data = df.drop(['Phase'], axis=1)
    n_components = 2  # Adjust this parameter
    n_neighbors = 15  # Adjust this parameter
    min_dist = 5   # Adjust this parameter
    spread = 5     # Adjust this parameter
    map_result, map_reducer = apply_umap(data, n_components, n_neighbors, min_dist, spread)

    draw_elbow(map_result, max_k=10, output_path='../output/dim_reduction/elbow_curve_umap_trans.png')

    num_clusters = 5
    map_labels, x_umap, y_umap, Z_umap, umap_cluster_centers_, inertia_ = apply_bi_kmeans_clustering(map_result,
                                                                                                      num_clusters)
    # Save index and cluster labels as csv file
    df = pd.DataFrame({'Phase': df['Phase'],
                       'Dim_1': map_result[:, 0],
                       'Dim_2': map_result[:, 1],
                       'Cluster': map_labels})
    df = pd.concat([df, data], axis=1)
    # Plot the results
    plot_results_pvt(df, map_labels, x_umap, y_umap, Z_umap, umap_cluster_centers_, inertia_, 'UMAP',
                 output_path='../output/dim_reduction/phase_vector_trans_umap_clustering_label.png')

    df.to_csv("../output/dim_reduction/phase_vector_trans_umap.csv", index=False, encoding='utf-8')

