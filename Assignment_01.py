import numpy as np
import spacy
import en_core_web_md
# import nevergrad
import random
# import sklearn
import os

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
    for label_type in set(y):
        print(f'{label_type} is {label_count}')
        for j in range(len(y)):
            if y[j] == label_type:
                y[j] = label_count
        label_count += 1
    y = np.array(y)
    return X, y


if __name__ == '__main__':
    path = './dataset'
    texts, labels = data_loader(path)
    X, y = data_vectorization(texts, labels)
    print(X.shape, y.shape)

