import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from collections import defaultdict

def distance(u, v):
    """Вычисление евклидова расстояния между векторами u и v"""
    return np.sqrt(np.sum((np.array(u, dtype=float) - np.array(v, dtype=float))**2))

def get_distances(xl, z):
    """Вычисление расстояний от точки z до всех точек в выборке"""
    if isinstance(xl, pd.DataFrame):
        return np.apply_along_axis(lambda row: distance(row, z), 1, xl.iloc[:, :-1])
    else:
        return np.array([distance(x, z) for x in xl])

def sort_objects_by_dist(xl, z):
    """Сортировка объектов по расстоянию до z"""
    features, labels = xl
    distances = get_distances(features, z)
    sorted_indices = np.argsort(distances)
    return features[sorted_indices], labels[sorted_indices]

def kNN(xl, z, k):
    """Алгоритм k ближайших соседей"""
    features, labels = xl
    sorted_features, sorted_labels = sort_objects_by_dist((features, labels), z)
    top_k_labels = sorted_labels[:k]
    unique, counts = np.unique(top_k_labels, return_counts=True)
    return unique[np.argmax(counts)]

def w_kwnn(i, k, q):
    """Весовая функция для kwNN"""
    return (i <= k) * (q ** i)

def kwNN(xl, z, k, q):
    """Алгоритм взвешенных k ближайших соседей"""
    features, labels = xl
    sorted_features, sorted_labels = sort_objects_by_dist((features, labels), z)
    weights = np.array([w_kwnn(i+1, k, q) for i in range(len(sorted_features))])
    sum_by_class = defaultdict(float)
    for cls, weight in zip(sorted_labels, weights):
        sum_by_class[cls] += weight
    return max(sum_by_class.items(), key=lambda x: x[1])[0]

def lOO(xl, method='knn'):
    """Метод скользящего контроля для подбора оптимальных параметров"""
    features, labels = xl
    l = len(features)
    if method == 'knn':
        lOOForK = np.zeros(l-1)
        for i in range(l):
            xl_minus_i_features = np.delete(features, i, axis=0)
            xl_minus_i_labels = np.delete(labels, i)
            ordered_features, ordered_labels = sort_objects_by_dist(
                (xl_minus_i_features, xl_minus_i_labels), features[i])
            max_k = min(len(ordered_features), len(lOOForK)+1)
            for v in range(1, max_k+1):
                if kNN((ordered_features, ordered_labels), features[i], v) != labels[i]:
                    lOOForK[v-1] += 1 / l
        return lOOForK
    else:  # kwNN
        q_range = np.arange(0.1, 1.1, 0.1)
        lOOForK = np.zeros((l-1, len(q_range)))
        for i in range(l):
            xl_minus_i_features = np.delete(features, i, axis=0)
            xl_minus_i_labels = np.delete(labels, i)
            ordered_features, ordered_labels = sort_objects_by_dist(
                (xl_minus_i_features, xl_minus_i_labels), features[i])
            max_k = min(len(ordered_features), len(lOOForK)+1)
            for v in range(1, max_k+1):
                for q_idx, q in enumerate(q_range):
                    if kwNN((ordered_features, ordered_labels), features[i], v, q) != labels[i]:
                        lOOForK[v-1, q_idx] += 1 / l
        return lOOForK

def getOptimalPar(lOOForK, method='knn'):
    """Нахождение оптимальных параметров"""
    if method == 'knn':
        return np.argmin(lOOForK) + 1, None
    else:  # kwNN
        min_idx = np.unravel_index(np.argmin(lOOForK), lOOForK.shape)
        return min_idx[0]+1, round((min_idx[1]+1)*0.1, 1)

def buildClassMap(xl, k, q=None, method='knn'):
    """Построение карты классификации"""
    classified_objects = []
    for i in np.arange(0, 7.1, 0.1):
        for j in np.arange(0, 2.6, 0.1):
            point = np.array([i, j], dtype=float)
            if method == 'knn':
                features, labels = xl
                cls = kNN((features, labels), point, k)
            else:  # kwNN
                features, labels = xl
                cls = kwNN((features, labels), point, k, q)
            classified_objects.append([i, j, cls])
    return pd.DataFrame(classified_objects, columns=['Petal.Length', 'Petal.Width', 'Species'])

def drawComparisonPlots(xl, knn_k, kwNN_k, kwNN_q, knn_lOO, kwNN_lOO, knn_classified, kwNN_classified):
    """Визуализация результатов для обоих методов"""
    colors = {"setosa": "red", "versicolor": "green", "virginica": "blue"}

    plt.figure(figsize=(18, 6))

    # Карта классификации kNN
    plt.subplot(1, 3, 1)
    features, labels = xl
    for species, color in colors.items():
        mask = (labels == species)
        plt.scatter(features[mask, 0], features[mask, 1], c=color, label=species)

    for species, color in colors.items():
        subset = knn_classified[knn_classified['Species'] == species]
        plt.scatter(subset['Petal.Length'], subset['Petal.Width'], c=color, marker='s', alpha=0.1)

    plt.title("Классификация с использованием kNN (k={})".format(knn_k))
    plt.xlabel("Длина лепестка")
    plt.ylabel("Ширина лепестка")
    plt.xlim(0, 7)
    plt.ylim(0, 2.5)
    plt.legend()

    # Карта классификации kwNN
    plt.subplot(1, 3, 2)
    features, labels = xl
    for species, color in colors.items():
        mask = (labels == species)
        plt.scatter(features[mask, 0], features[mask, 1], c=color, label=species)

    for species, color in colors.items():
        subset = kwNN_classified[kwNN_classified['Species'] == species]
        plt.scatter(subset['Petal.Length'], subset['Petal.Width'], c=color, marker='s', alpha=0.1)

    plt.title("Классификация с использованием kwNN (k={}, q={})".format(kwNN_k, kwNN_q))
    plt.xlabel("Длина лепестка")
    plt.ylabel("Ширина лепестка")
    plt.xlim(0, 7)
    plt.ylim(0, 2.5)
    plt.legend()

    # Графики LOO
    plt.subplot(1, 3, 3)

    # kNN LOO
    plt.plot(range(1, len(knn_lOO)+1), knn_lOO, 'r-', label='kNN')
    plt.scatter([knn_k], [knn_lOO[knn_k-1]], c='red')
    label_knn = f"kNN: k = {knn_k}\nLOO = {round(knn_lOO[knn_k-1], 3)}"
    plt.text(knn_k, knn_lOO[knn_k-1], label_knn, ha='center', va='bottom')

    # kwNN LOO (для оптимального q)
    q_idx = int(kwNN_q * 10) - 1
    plt.plot(range(1, len(kwNN_lOO)+1), kwNN_lOO[:, q_idx], 'b-', label='kwNN')
    plt.scatter([kwNN_k], [kwNN_lOO[kwNN_k-1, q_idx]], c='blue')
    label_kwnn = f"kwNN: k = {kwNN_k}, q = {kwNN_q}\nLOO = {round(kwNN_lOO[kwNN_k-1, q_idx], 3)}"
    plt.text(kwNN_k, kwNN_lOO[kwNN_k-1, q_idx], label_kwnn, ha='center', va='top')

    plt.title("Сравнение LOO")
    plt.xlabel("k Значения")
    plt.ylabel("LOO Значения")
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    """Основная функция"""
    iris = load_iris()
    features = iris.data[:, 2:4]
    labels = iris.target_names[iris.target]

    print("Расчет для kNN...")
    knn_lOO = lOO((features, labels), 'knn')
    knn_k, _ = getOptimalPar(knn_lOO, 'knn')

    print("Расчет для kwNN...")
    kwNN_lOO = lOO((features, labels), 'kwNN')
    kwNN_k, kwNN_q = getOptimalPar(kwNN_lOO, 'kwNN')

    print(f"Оптимальные параметры для kNN: k = {knn_k}")
    print(f"Оптимальные параметры для kwNN: k = {kwNN_k}, q = {kwNN_q}")

    print("Карты классификации...")
    knn_classified = buildClassMap((features, labels), knn_k, method='knn')
    kwNN_classified = buildClassMap((features, labels), kwNN_k, kwNN_q, method='kwNN')

    drawComparisonPlots((features, labels), knn_k, kwNN_k, kwNN_q, knn_lOO, kwNN_lOO,
                      knn_classified, kwNN_classified)

if __name__ == "__main__":
    main()
