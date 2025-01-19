import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns


def visualize_clusters(data, labels, method="PCA", perplexity=30, n_components=2, png_name='visualization.png'):
    """
    Визуализация кластеров в 2D или 3D.
    
    Параметры:
    - data: np.array, эмбеддинги.
    - labels: np.array, метки кластеров.
    - method: "PCA" или "TSNE", метод снижения размерности.
    - perplexity: параметр для t-SNE (используется только для t-SNE).
    - n_components: размерность (2 или 3).
    """
    if method == 'PCA':
        reducer = PCA(n_components=n_components)
    elif method == 'TSNE':
        reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    else:
        raise ValueError("Метод должен быть 'PCA' или 'TSNE'")
    
    reduced_data = reducer.fit_transform(data)
    
    if n_components == 2:
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x=reduced_data[:, 0],
            y=reduced_data[:, 1],
            hue=labels,
            palette='viridis',
            s=50,
            alpha=0.8
        )
        plt.title(f'Визуализация кластеров ({method})')
        plt.xlabel('Компонента 1')
        plt.ylabel('Компонента 2')
        plt.legend(title='Кластеры')
        plt.savefig('./figs/' + f'2d__{method}_' + png_name, dpi=300)
        plt.show()

    elif n_components == 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(
            reduced_data[:, 0],
            reduced_data[:, 1],
            reduced_data[:, 2],
            c=labels,
            cmap='viridis',
            s=50,
            alpha=0.8
        )
        ax.set_title(f'Визуализация кластеров ({method})')
        ax.set_xlabel('Компонента 1')
        ax.set_ylabel('Компонента 2')
        ax.set_zlabel('Компонента 3')
        fig.colorbar(scatter, ax=ax, label='Кластеры')
        plt.savefig('./figs/' + f'3d_{method}_' + png_name, dpi=300)
        plt.show()
