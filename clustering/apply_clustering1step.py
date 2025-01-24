import argparse
import os
import numpy as np
import pandas as pd
import clustering
from vis_clusters import visualize_clusters
from kmeans import ApplyKmeans

from dotenv import load_dotenv

load_dotenv()
SSD_PATH = os.getenv('SSD_PATH')
DATASET_NAME = os.getenv('DATASET_NAME')
SEGMENT_EMB_DIR = os.getenv('SEGMENT_EMB_DIR')


def main(speaker_name, max_clusters, min_clusters, cluster_range):
    segment_embeddings = clustering.load_and_average_embeddings(
        os.path.join(SSD_PATH, DATASET_NAME, SEGMENT_EMB_DIR),
        speaker_name=speaker_name,
    )
    print(f'Загружено {len(segment_embeddings)} сегментов')

    all_embeddings = clustering.prepare_data_for_clustering(segment_embeddings)
    print(f'Размер данных для кластеризации: {all_embeddings.shape}')

    cluster_range, distortions, silhouette_scores, kmeans_models = clustering.find_optimal_clusters(
        all_embeddings,
        min_clusters=min_clusters,
        max_clusters=max_clusters,
        cluster_range=cluster_range,
    )

    clustering.plot_cluster_metrics(
        cluster_range, distortions,
        silhouette_scores,
        png_name=f'metrics_1step_clustering_{max_clusters}_clusters_{speaker_name}sp.png',
    )

    optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
    print(f'Оптимальное количество кластеров (по силуэту): {optimal_clusters}')

    model = ApplyKmeans(
        km_path=f'{SSD_PATH}/models/best_kmeans_model_{optimal_clusters}.joblib',
        use_gpu=False,
    )

    df = pd.DataFrame({
        'segment_name': list(segment_embeddings.keys()),
        'mean_embedding': list(segment_embeddings.values())
    })

    segments_labels = model(all_embeddings)
    df['1step_cluster_label'] = segments_labels.tolist()

    df[['segment_name', '1step_cluster_label']].to_csv(f'./annotation_segments_{optimal_clusters}_{speaker_name}sp.txt', index=False)

    visualize_clusters(
        all_embeddings,
        segments_labels.tolist(),
        method="TSNE",
        perplexity=30,
        n_components=2,
        png_name=f'1step_clustering_{optimal_clusters}_{speaker_name}sp.png',
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clustering script for audio segments.')
    parser.add_argument('--speaker_name', type=str, required=True, help='Name of the speaker')
    parser.add_argument('--max_clusters', type=int, required=True, help='Maximum number of clusters')
    parser.add_argument('--min_clusters', type=int, required=True, help='Minimum number of clusters')
    parser.add_argument('--cluster_range', type=int, required=True, help='Range number of clusters')

    args = parser.parse_args()
    main(args.speaker_name, args.max_clusters, args.min_clusters, args.cluster_range)
