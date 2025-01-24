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


def main(speaker_name, max_clusters, min_clusters, cluster_range, step2_clusters=494, batch_size=64, optimal_clusters=0):
    segment_embeddings = clustering.load_and_average_embeddings(
        os.path.join(SSD_PATH, DATASET_NAME, SEGMENT_EMB_DIR),
        speaker_name=speaker_name,
    )
    print(f'Загружено {len(segment_embeddings)} сегментов')

    all_embeddings = clustering.prepare_data_for_clustering(segment_embeddings)
    print(f'Размер данных для кластеризации: {all_embeddings.shape}')

    if optimal_clusters == 0:
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

    if not os.path.exists(f'./annotation_segments_{optimal_clusters}_{speaker_name}sp.txt'):
        df = pd.DataFrame({
            'segment_name': list(segment_embeddings.keys()),
            'mean_embedding': list(segment_embeddings.values())
        })
        columns = ['segment_name']
    else:
        df = pd.read_csv(f'./annotation_segments_{optimal_clusters}_{speaker_name}sp.txt')
        columns = list(df.columns())

    quantizer1, quant2_segments_labels = clustering.cluster_embeddings_in2steps(
        all_embeddings,
        n_clusters_step1=optimal_clusters,
        n_clusters_step2=step2_clusters,
        seed=42,
        km_path=None,
        save_dir=f'{SSD_PATH}/best_kmeans_model_{optimal_clusters}.joblib',
    )

    cluster1 = quantizer1(all_embeddings) if isinstance(quantizer1, ApplyKmeans) else quantizer1.predict(np.array(all_embeddings))
    segments_labels = quant2_segments_labels[cluster1]

    df['2step_cluster_label'] = segments_labels.tolist()
    columns = columns + ['2step_cluster_label']

    df[list(set(columns))].to_csv(f'./annotation_segments_{optimal_clusters}_{speaker_name}sp.txt', index=False)

    visualize_clusters(
        all_embeddings,
        segments_labels.tolist(),
        method="TSNE",
        perplexity=30,
        n_components=2,
        png_name=f'2step_clustering_{optimal_clusters}_{step2_clusters}_{speaker_name}sp.png',
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clustering script for audio segments.')
    parser.add_argument('--speaker_name', type=str, default='', required=False, help='Name of the speaker')
    parser.add_argument('--max_clusters', type=int, default=4096, required=False, help='Maximum number of clusters')
    parser.add_argument('--min_clusters', type=int, default=44, required=False, help='Minimum number of clusters')
    parser.add_argument('--cluster_range', type=int, default=50, required=False, help='Range number of clusters')
    parser.add_argument('--step2_clusters', type=int, required=True, help='Range number of clusters')
    parser.add_argument('--batch_size', type=int, default=64, required=False, help='Range number of clusters')
    parser.add_argument('--optimal_clusters', type=int, default=0, required=False, help='Range number of clusters')

    args = parser.parse_args()
    main(
        args.speaker_name,
        args.max_clusters,
        args.min_clusters,
        args.cluster_range,
        step2_clusters=args.step2_clusters,
        batch_size=args.batch_size,
        optimal_clusters=args.optimal_clusters,
    )
