import numpy as np
import spacy
import nevergrad
import random
import sklearn
import os

# Question 0
student_id = 22406581
random.seed(student_id)
np.random.seed(student_id)


# Question 1
def data_loader(path):
    texts = []
    labels = []
    for label_name in os.listdir(path):
        labels.append(label_name)
        for text_name in os.listdir(path + '/' + label_name):
            with open('{}/{}/{}'.format(path, label_name, text_name), 'r') as file:
                texts.append(file.readlines())
                # print(texts[0][0])
                # break
        # break
    return texts, labels


if __name__ == '__main__':
    path = './dataset'
    texts, labels = data_loader(path)
    print(len(texts), labels)
