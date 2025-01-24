import os
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import joblib


def load_and_average_embeddings(directory):
    """
    Загружает эмбеддинги из папок и усредняет их для каждого сегмента.
    Возвращает словарь: {название_сегмента: усредненный эмбеддинг}.
    """
    segment_embeddings = {}
    for audio_name in os.listdir(directory):
        audio_path = os.path.join(directory, audio_name)

        segments_in_audio = os.listdir(audio_path)
        for segment in segments_in_audio:
          embedding_path = os.path.join(audio_path, segment)
          embeddings = np.load(embedding_path)
          average_embedding = embeddings.mean(axis=1).squeeze(0)
          segment_embeddings[segment.replace('.npy', '')] = average_embedding

    return segment_embeddings


def prepare_data_for_clustering(segment_embeddings):
    """
    Объединяет все усредненные эмбеддинги из всех аудио в единый массив для кластеризации.
    """
    all_embeddings = []
    for embedding in segment_embeddings.values():
        all_embeddings.append(embedding)
    return np.array(all_embeddings)


def find_optimal_clusters(data, min_clusters=2, max_clusters=15, cluster_range=10):
    """
    Ищет оптимальное количество кластеров с использованием метода локтя и коэффициента силуэта.
    """
    print(data.shape)
    distortions = []
    silhouette_scores = []

    best_silhouette_score = -1
    best_model = None

    cluster_range = range(min_clusters, max_clusters + cluster_range, cluster_range)
    models = []
    for n_clusters in cluster_range:
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=32, random_state=42)
        kmeans.fit(data)
        distortions.append(kmeans.inertia_)
        current_silhouette_score = silhouette_score(data, kmeans.labels_)
        silhouette_scores.append(current_silhouette_score)
        models.append(kmeans)

        inertia = -kmeans.score(data) / len(data)
        print(f'kmeans for {n_clusters} total intertia: {inertia:.5f}')

        if current_silhouette_score > best_silhouette_score:
            best_silhouette_score = current_silhouette_score
            best_model = kmeans
            joblib.dump(best_model, f'best_kmeans_model_{n_clusters}.joblib')
            print(f'New best model saved with silhouette score: {best_silhouette_score:.4f} for {n_clusters} clusters.')

    return cluster_range, distortions, silhouette_scores, models


def plot_cluster_metrics(cluster_range, distortions, silhouette_scores, png_name='metrics_vis.png'):
    """
    Строит графики для метода локтя и коэффициента силуэта.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].plot(cluster_range, distortions, marker='o')
    ax[0].set_title('Метод локтя')
    ax[0].set_xlabel('Количество кластеров')
    ax[0].set_ylabel('Инерция (Distortion)')

    ax[1].plot(cluster_range, silhouette_scores, marker='o', color='orange')
    ax[1].set_title('Коэффициент силуэта')
    ax[1].set_xlabel('Количество кластеров')
    ax[1].set_ylabel('Силуэт')

    plt.tight_layout()
    plt.savefig('./figs/' + png_name, dpi=300)
    plt.show()


def get_cluster_segments(model, embeddings):
    """
    Выполняет кластеризацию сегментов с заданным количеством кластеров.
    """
    if len(embeddings.shape) == 1:
        embeddings = np.array([embeddings])

    assert len(embeddings.shape) == 2

    labels = model.fit_predict(embeddings)
    return labels
