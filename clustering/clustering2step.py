from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
import numpy as np
import joblib
from pathlib import Path


def cluster_embeddings_in2steps(embeddings, batch_size=10000, n_clusters_step1=500, n_clusters_step2=100, seed=42, save_dir='clusters'):

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    quantizer1 = MiniBatchKMeans(
        n_clusters=n_clusters_step1,
        batch_size=batch_size,
        verbose=1,
        compute_labels=False,
        random_state=seed,
        max_no_improvement=100,
        n_init=5,
        reassignment_ratio=0.0,
    )
    quantizer1.fit(embeddings)

    joblib.dump(quantizer1, save_dir / 'quantizer1.joblib')

    quantizer2 = AgglomerativeClustering(n_clusters=n_clusters_step2)
    cluster_labels = quantizer2.fit_predict(quantizer1.cluster_centers_)

    joblib.dump(quantizer2, save_dir / 'quantizer2.joblib')

    np.save(save_dir / 'quantizer2_labels.npy', cluster_labels)

    return quantizer1, cluster_labels


def get_cluster(quantizer1, quantizer2_labels, embeddings):

    if len(embeddings.shape) == 1:
        embeddings = np.array([embeddings])

    assert len(embeddings.shape) == 2

    cluster1 = quantizer1.predict(np.array(embeddings))
    cluster2 = quantizer2_labels[cluster1]
    return cluster2
