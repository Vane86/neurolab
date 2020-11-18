import csv
from scipy import fft
from matplotlib import pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


def concat_csvs(file_pathes, result_path):
    with open(result_path, mode='w') as output_file:
        for i, file_path in enumerate(file_pathes):
            with open(file_path) as input_file:
                header = next(input_file)
                if i == 0:
                    output_file.write(header)
                for line in input_file:
                    output_file.write(line)


def load_data(file_path):
    l2c = list()  # label to class mark
    features, labels = list(), list()
    with open(file_path) as input_file:
        reader = csv.reader(input_file, delimiter=',')
        header = next(reader)
        for row in reader:
            features.append([float(x) for x in row[2:]])
            if row[0] not in l2c:
                l2c.append(row[0])
            labels.append(l2c.index(row[0]))
    return np.array(features), np.array(labels)


def extract_features(raw_features, labels, window_size=128, step_factor=0.5):
    step = int(window_size * step_factor)
    new_features, new_labels = list(), list()
    for i in range(0, raw_features.shape[0] - window_size, step):
        if len(set(labels[i:i + window_size])) != 1:
            continue
        channel_ffts = list()
        for j in range(raw_features.shape[1]):
            window = raw_features[i:i + window_size, j]
            win_mean = window.mean()
            channel_ffts.extend([abs(x) for x in fft(window - win_mean)[:window_size // 2]])
        new_features.append(channel_ffts)
        new_labels.append(labels[i])
    return np.array(new_features), np.array(new_labels)


features, labels = extract_features(*load_data('data/new_data.csv'))
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2)

# pca = PCA(n_components=features_train.shape[1] // 10).fit(features_train)
# features_train = pca.transform(features_train)
# features_test = pca.transform(features_test)
# print(sum(pca.explained_variance_ratio_))

print(features_train.shape, len(features_test))

clf = MLPClassifier(max_iter=300).fit(features_train, labels_train)
print('Train score: ', clf.score(features_train, labels_train))
print('Test score: ', clf.score(features_test, labels_test))

# concat_csvs(('data/back.csv', 'data/forward.csv', 'data/left.csv', 'data/neutral.csv', 'data/right.csv'), 'data/data.csv')
