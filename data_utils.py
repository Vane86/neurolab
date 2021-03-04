import csv

import numpy as np
import torch
from scipy import fft

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score


def concat_csvs(file_pathes, result_path):
    with open(result_path, mode='w') as output_file:
        for i, file_path in enumerate(file_pathes):
            with open(file_path) as input_file:
                header = next(input_file)
                if i == 0:
                    output_file.write(header)
                for line in input_file:
                    output_file.write(line)


def load_data(file_path, label2class=None):
    l2c = label2class or list()  # label to class mark
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


def johnson_transform(data, data_len_factor=0.75):
    result = list()
    for i in range(int(len(data) * data_len_factor)):
        result.append(sum(abs(data[j + i] - data[j]) for j in range(len(data) - i)) / (len(data) - i))
    return np.array(result)


def divide_windows(features, labels, window_size, step_factor):
    step = int(window_size * step_factor)
    new_features, new_labels = list(), list()
    for i in range(0, features.shape[0] - window_size + 1, step):
        if len(set(labels[i:i + window_size])) != 1:
            continue
        new_features.append([features[i:i + window_size, j] for j in range(features.shape[1])])
        new_labels.append(labels[i])
    return np.array(new_features), np.array(new_labels)


def avg_windows(features, labels, n_windows, step_windows):
    new_features, new_labels = list(), list()
    for i in range(0, features.shape[0] - n_windows + 1, step_windows):
        if len(set(labels[i:i + n_windows])) != 1:
            continue
        new_features.append([np.sum(features[i:i + n_windows, j, :], axis=0) / n_windows for j in range(features.shape[1])])
        new_labels.append(labels[i])
    return np.array(new_features), np.array(new_labels)


def fourier_windows(features, labels, freq_ignore_factor=0.0):
    new_features, new_labels = list(), list()
    for i in range(features.shape[0]):
        row = list()
        for j in range(features.shape[1]):
            row.append([np.log1p(abs(x)) for x in fft(features[i, j, :] - features[i, j, :].mean())[:int((1 - freq_ignore_factor) * features.shape[2] / 2)]])
        new_features.append(row)
        new_labels.append(labels[i])
    return np.array(new_features), np.array(new_labels)


def johnson_windows(features, labels):
    new_features, new_labels = list(), list()
    for i in range(features.shape[0]):
        row = [johnson_transform(features[i, j, :]) for j in range(features.shape[1])]
        new_features.append(row)
        new_labels.append(labels[i])
    return np.array(new_features), np.array(new_labels)


def differential_windows(features, labels):
    new_features, new_labels = list(), list()
    for i in range(features.shape[0]):
        row = [(np.roll(features[i, j, :], 1) - np.roll(features[i, j, :], -1))[1:-1] / 2 for j in range(features.shape[1])]
        new_features.append(row)
        new_labels.append(labels[i])
    return np.array(new_features), np.array(new_labels)


def concat_by_channels(features, labels):
    return features.reshape(features.shape[0], features.shape[1] * features.shape[2]), labels


def plot_windows(features, labels, only=None):

    borders_xs = list()
    for i in range(len(labels) - 1):
        if labels[i + 1] - labels[i] != 0:
            borders_xs.append(i)

    iterable = only or range(features.shape[1])

    figure, axes = plt.subplots(len(iterable), 1, sharex=True)

    for i, index in enumerate(iterable):
        im = axes[i].imshow(features[:, index, :].T)
        for line_x in borders_xs:
            axes[i].axvline(x=line_x, color='red')
        axes[i].figure.colorbar(im, ax=axes[i])

    figure.tight_layout()
    figure.subplots_adjust(hspace=0)
    plt.show()


def pipeline(classifier, stages, train_data_file_path, test_data_file_path, data_type='array', label2class=None, features_processors=None, features_selectors=None):
    features, labels = load_data(train_data_file_path, label2class)

    for stage in stages:
        features, labels = stage[0](features, labels, **stage[1])

    features_train, features_val, labels_train, labels_val = train_test_split(features, labels, test_size=0.15)

    if features_processors:
        for processor in features_processors:
            processor.fit(features_train)
            features_train = processor.transform(features_train)
            features_val = processor.transform(features_val)
            print(f'Train features shape after {processor}: {features_train.shape}')

    if features_selectors:
        for selector in features_selectors:
            selector.fit(features_train, labels_train)
            features_train = selector.transform(features_train)
            features_val = selector.transform(features_val)
            print(f'Train features shape after {selector}: {features_train.shape}')

    if data_type == 'tensor':
        features_train, labels_train = torch.tensor(features_train, dtype=torch.float32), torch.tensor(labels_train, dtype=torch.long)
        features_val, labels_val = torch.tensor(features_val, dtype=torch.float32), torch.tensor(labels_val, dtype=torch.long)
        classifier.fit(features_train, labels_train)

    print('Train score:', f1_score(classifier.predict(features_train), labels_train, average='weighted'))
    print('Validation score:', f1_score(classifier.predict(features_val), labels_val, average='weighted'))

    features_test, labels_test = load_data(test_data_file_path, label2class)

    for stage in stages:
        features_test, labels_test = stage[0](features_test, labels_test, **stage[1])

    if features_processors:
        for processor in features_processors:
            features_test = processor.transform(features_test)

    if features_selectors:
        for selector in features_selectors:
            features_test = selector.transform(features_test)

    if data_type == 'tensor':
        features_test, labels_test = torch.tensor(features_test, dtype=torch.float32), torch.tensor(labels_test, dtype=torch.long)
    print('Test score:', f1_score(classifier.predict(features_test), labels_test, average='weighted'))


if __name__ == '__main__':
    # test_features = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    # test_labels = [1, 1, 2, 2]
    # windows, labels = divide_windows(test_features, test_labels, 2, 1.0)
    # print(windows, labels)
    # print()
    # avg, labels = avg_windows(windows, [1, 1], 2, 1)
    # print(avg, labels)
    # print()
    # fourier, labels = fourier_windows(avg, labels)
    # print(fourier, labels)
    # print()
    # channels, labels = concat_by_channels(windows, [1, 2])
    # print(channels, labels)
    #
    # sine = np.array([np.sin(10 * np.linspace(-10, 10, 500))]).T
    # f, l = divide_windows(sine, [1] * len(sine), window_size=50, step_factor=1.0)
    # f, l = fourier_windows(f, l)
    # plot_windows(f[:, 0, :])
    #
    # noisy_sine = np.array([np.sin(10 * np.linspace(-1000, 1000, 10000)) + np.random.normal(0, 1, 10000)]).T
    # f, l = divide_windows(noisy_sine, [1] * len(noisy_sine), window_size=50, step_factor=1.0)
    # f_avg, l_avg = avg_windows(f, l, n_windows=20, step_windows=20)
    # f_f, l_f = fourier_windows(f, l)
    # f_af, l_af = fourier_windows(f_avg, l_avg)
    # plot_windows(f_f[:, 0, :])
    # plot_windows(f_af[:, 0, :])

    # x = np.linspace(0, 100, 200)
    # sine = np.sin(x)
    # jt = johnson_transform(sine)
    # plt.plot(x, sine)
    # plt.plot(jt)
    # jt_diff = (np.roll(jt, 1) - np.roll(jt, -1))[1:-1] / 2
    # plt.plot(jt_diff)
    # plt.show()

    pass
