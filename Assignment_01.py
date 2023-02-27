import matplotlib.pyplot as plt
import numpy as np
import os
import en_core_web_md
import nevergrad
import random
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
from sklearn.metrics import silhouette_score
from sklearn.metrics import accuracy_score
from collections import Counter
import warnings

warnings.filterwarnings('ignore')


# Question 0
student_id = 22406581
random.seed(student_id)
np.random.seed(student_id)


# Question 1
def data_loader(path):
    texts = []
    labels = []
    texts_reduce = []
    labels_reduce = []
    for label_name in os.listdir(path):
        for text_name in os.listdir(path + '/' + label_name):
            with open('{}/{}/{}'.format(path, label_name, text_name), 'r', encoding='utf-8') as file:
                text = file.read()
                texts.append(text)
            labels.append(label_name)
    # reduce to 120 per label
    reduce_120 = np.random.randint(0, 150, 120).tolist() + np.random.randint(150, 300, 120).tolist() + np.random.randint(300, 450, 120).tolist()
    for i in reduce_120:
        texts_reduce.append(texts[i])
        labels_reduce.append(labels[i])

    return texts_reduce, labels_reduce


# Question 2
def data_vectorization(texts, labels):
    nlp = en_core_web_md.load()
    docs = [nlp(text) for text in texts]
    X = np.vstack([doc.vector for doc in docs])
    y = labels.copy()
    label_count = 0
    label = []
    print('Transforming labels into integer')
    for label_type in set(y):
        print(f'{label_type} is {label_count}')
        label.append(label_type)
        for j in range(len(y)):
            if y[j] == label_type:
                y[j] = label_count
        label_count += 1
    y = np.array(y)
    return X, y, label


# Question 3
def data_visualization(X, y, label):
    pca = PCA(n_components=2)
    X_transformed = pca.fit_transform(X)
    plt.scatter(x=X_transformed[:, 0], y=X_transformed[:, 1], c=y)
    plt.title('2D scatter plot')
    plt.xlabel('Principle component 1')
    plt.ylabel('Principle component 2')
    plt.show()
    return True


# Question 4
def classification(X, y):
    # Define classifiers
    classifiers = {
        'LR': LogisticRegression(),
        'RF': RandomForestClassifier(n_estimators=10, max_depth=6),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    # Define k-fold cross-validation
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5)
    # Define the PCA
    pca = PCA(n_components=0.95)
    lr_accs = []
    rf_accs = []
    knn_accs = []
    # Splitting, training and testing
    for train_idx, test_idx in rskf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        for classifier_name, classifier in classifiers.items():
            pipeline = make_pipeline(pca, classifier)
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            # testing
            acc = accuracy_score(y_test, y_pred)
            if classifier_name == 'LR':
                lr_accs.append(acc)
            elif classifier_name == 'RF':
                rf_accs.append(acc)
            elif classifier_name == 'KNN':
                knn_accs.append(acc)
    print('LR:')
    print('Mean accuracy:', np.mean(lr_accs))
    print('Standard deviation:', np.std(lr_accs))
    print('-'*50)
    print('RF:')
    print('Mean accuracy:', np.mean(rf_accs))
    print('Standard deviation:', np.std(rf_accs))
    print('-'*50)
    print('KNN:')
    print('Mean accuracy:', np.mean(knn_accs))
    print('Standard deviation:', np.std(knn_accs))
    return True


# Question 5
def dimensionality_reduction(X):
    def decline_components(X):
        pca = PCA()
        X_transformed = pca.fit_transform(X)
        explained_variance = pca.explained_variance_ratio_
        components = 0
        variance_num = 0
        while variance_num < 0.95:
            variance_num += explained_variance[components]
            components += 1
        print(components, variance_num)
        return components
    components = decline_components(X)
    pca = PCA(n_components=components)
    X_transformed = pca.fit_transform(X)
    return X_transformed


# Question 6
def cluster(X, y):
    # normalize the data-matrix
    normalizer = Normalizer()
    X = normalizer.fit_transform(X)
    for k in range(2, 5):
        kmeans = KMeans(n_clusters=k, random_state=4)
        kmeans.fit(X)
        # Define the inertia, the smaller, the better
        inertia = kmeans.inertia_
        silhouette = silhouette_score(X, kmeans.labels_)
        print(f"k={k}:")
        print(f"Inertia: {inertia:.2f}")
        print(f"Silhouette score: {silhouette:.2f}\n")
        print("-"*50)
    return True


# Question 7
def validation(token):
    return not token.is_stop and (token.pos_ == 'NOUN' or token.pos_ == 'PROPN')


def cluster_information(texts, X):
    nlp = en_core_web_md.load()
    all_lemmas = []
    for text in texts:
        doc = nlp(text)
        all_lemmas += [token.lemma_ for token in doc if validation(token)]
    # clustering
    km = KMeans(n_clusters=3, random_state=22449515)
    km.fit(X)
    labels = km.labels_
    # Storing 10 most common lemmas of each label
    all_lemma = []
    for i in range(3):
        lemma_10 = []
        indices = [j for j in range(len(labels)) if labels[j] == i]
        cluster_lemmas = []
        for index in indices:
            doc = nlp(texts[index])
            cluster_lemmas += [token.lemma_ for token in doc if validation(token)]
        lemma_counts = Counter(cluster_lemmas)
        print(f"Cluster {i}")
        for lemma, count in lemma_counts.most_common(10):
            lemma_10.append(lemma)
            print(f"{lemma}: {count}")
        print("-"*50)
        all_lemma.append(lemma_10)
    return all_lemma


# Question 8
def cluster_interpretation(lemmas):
    cluster_labels = []
    for lemma_10 in lemmas:
        if 'stingray' in lemma_10:
            label = 'animals'
        elif 'plant' in lemma_10:
            label = 'fruit'
        elif 'country' in lemma_10:
            label = 'geopolitics'
        cluster_labels.append(label)
    for id, label in enumerate(cluster_labels):
        print(f"Cluster {id}: {label}")
    return True


if __name__ == '__main__':
    print("-" * 50)
    # question 1
    path = './dataset'
    texts, labels = data_loader(path)
    # question 2
    X, y, label = data_vectorization(texts, labels)
    print("-" * 50)
    print('After reducing, data array shape is {}, and label array shape is {}'.format(X.shape, y.shape))
    print("-" * 50)
    # question 3
    data_visualization = data_visualization(X, y, label)
    # question 4
    classification = classification(X, y)
    # question 5
    X_transformed = dimensionality_reduction(X)
    print("-" * 50)
    print('After PCA the shape of data is {}'.format(X_transformed.shape))
    print("-" * 50)
    # question 6
    cluster = cluster(X, y)
    # question 7
    lemmas = cluster_information(texts, X)
    # question 8
    cluster_interpretation = cluster_interpretation(lemmas)


