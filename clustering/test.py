import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import joblib


# === 1. Загрузка эмбеддингов и усреднение ===
def load_and_average_embeddings(directory, sample_rate):
    """
    Загружает эмбеддинги из папок и усредняет их для каждого сегмента.
    Возвращает словарь: {название_сегмента: усредненный эмбеддинг}.
    """
    audio_embeddings = {}
    segment_embeddings = {}
    total = 0
    for audio_name in os.listdir(directory):
        audio_path = os.path.join(directory, audio_name)

        segments_in_audio = os.listdir(audio_path)
        for segment in segments_in_audio:
          embedding_path = os.path.join(audio_path, segment)
          embeddings = np.load(embedding_path)
          average_embedding = embeddings.mean(axis=1).squeeze(0)
          segment_embeddings[segment.replace('.npy', '')] = average_embedding

    return segment_embeddings


# === 2. Преобразование данных для кластеризации ===
def prepare_data_for_clustering(segment_embeddings):
    """
    Объединяет все усредненные эмбеддинги из всех аудио в единый массив для кластеризации.
    """
    all_embeddings = []
    for embedding in segment_embeddings.values():
        all_embeddings.append(embedding)
    return np.array(all_embeddings)

# === 3. Вывод статистики по длительностям ===
def print_duration_statistics(durations):
    """
    Выводит статистику по длительности сегментов.
    """
    print(f"Количество сегментов: {len(durations)}")
    print(f"Средняя длительность сегмента: {np.mean(durations):.6f} секунд")
    print(f"Медианная длительность сегмента: {np.median(durations):.6f} секунд")
    print(f"Минимальная длительность сегмента: {np.min(durations):.6f} секунд")
    print(f"Максимальная длительность сегмента: {np.max(durations):.6f} секунд")

# === 4. Подбор оптимального количества кластеров ===
def find_optimal_clusters(data, max_clusters=15):
    """
    Ищет оптимальное количество кластеров с использованием метода локтя и коэффициента силуэта.
    """
    print(data.shape)
    distortions = []
    silhouette_scores = []
    cluster_range = range(2, max_clusters + 1)
    
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(data)
        distortions.append(kmeans.inertia_)  # Сумма квадратов расстояний до центроидов
        silhouette_scores.append(silhouette_score(data, kmeans.labels_))
    
    return cluster_range, distortions, silhouette_scores

# === 5. Построение графиков ===
def plot_cluster_metrics(cluster_range, distortions, silhouette_scores):
    """
    Строит графики для метода локтя и коэффициента силуэта.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Метод локтя
    ax[0].plot(cluster_range, distortions, marker='o')
    ax[0].set_title("Метод локтя")
    ax[0].set_xlabel("Количество кластеров")
    ax[0].set_ylabel("Инерция (Distortion)")

    # Коэффициент силуэта
    ax[1].plot(cluster_range, silhouette_scores, marker='o', color='orange')
    ax[1].set_title("Коэффициент силуэта")
    ax[1].set_xlabel("Количество кластеров")
    ax[1].set_ylabel("Силуэт")

    plt.tight_layout()
    plt.show()

# === 6. Кластеризация сегментов ===
def cluster_segments(data, n_clusters):
    """
    Выполняет кластеризацию сегментов с заданным количеством кластеров.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data)
    joblib.dump(kmeans, f'./kmeans_{n_clusters}.pkl')
    return labels, kmeans

# === 7. Основной эксперимент ===
def main_experiment(directory, sample_rate, max_clusters=1024):
    # 1. Загрузка эмбеддингов и усреднение
    segment_embeddings = load_and_average_embeddings(directory, sample_rate)
    print(f"Загружено {len(segment_embeddings)} сегментов")


    # # 2. Подготовка данных для кластеризации
    all_embeddings = prepare_data_for_clustering(segment_embeddings)
    print(f"Размер данных для кластеризации: {all_embeddings.shape}")

    # # 3. Статистика по длительности сегментов
    # print_duration_statistics(durations)

    # print(all_embeddings.shape)

    # # 4. Поиск оптимального количества кластеров
    cluster_range, distortions, silhouette_scores = find_optimal_clusters(all_embeddings, max_clusters)
    plot_cluster_metrics(cluster_range, distortions, silhouette_scores)

    # # 5. Выбор количества кластеров (например, по графику локтя или максимуму силуэта)
    optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
    print(f"Оптимальное количество кластеров (по силуэту): {optimal_clusters}")

    # # 6. Кластеризация с оптимальным количеством кластеров
    # labels, kmeans = cluster_segments(all_embeddings, optimal_clusters)
    # print(f"Кластеризация завершена. Количество кластеров: {optimal_clusters}")

    # return audio_embeddings, all_embeddings, durations, labels, kmeans

# # === Запуск эксперимента ===
if __name__ == "__main__":
    directory = "/content/librispeech_asr_dummy/hubert_embeddings/"
    sample_rate = 16000
    main_experiment(directory, sample_rate)

