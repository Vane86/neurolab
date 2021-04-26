import csv

import numpy as np
import torch
from scipy.fft import fft

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
import random

from pyts.approximation import SymbolicAggregateApproximation


def concat_csvs(file_pathes, result_path):
    with open(result_path, mode='w') as output_file:
        for i, file_path in enumerate(file_pathes):
            with open(file_path) as input_file:
                header = next(input_file)
                if i == 0:
                    output_file.write(header)
                for line in input_file:
                    output_file.write(line)


def __squeeze(lst):
    result = list()
    for el in lst:
        result.extend(el)
    return result


def load_data(file_path, test_size=0.25, channels=None, label2class=None):
    l2c = label2class or list()  # label to class mark
    # features, labels = list(), list()
    features, labels = list(), list()
    with open(file_path) as input_file:
        reader = csv.reader(input_file, delimiter=',')
        header = next(reader)

        features_chunk, labels_chunk = list(), list()
        last_label = None

        for row in reader:
            row_channels = row[1:]
            if channels is not None:
                row_channels = [row_channels[i] for i in channels]
            if last_label is None:
                last_label = row[0]
            features_chunk.append([float(x) for x in row_channels])
            if row[0] not in l2c:
                l2c.append(row[0])
            labels_chunk.append(l2c.index(row[0]))
            if last_label != row[0]:
                features.append(features_chunk[:-1])
                labels.append(labels_chunk[:-1])
                features_chunk, labels_chunk = [features_chunk[-1]], [labels_chunk[-1]]
                last_label = row[0]
        features.append(features_chunk)
        labels.append(labels_chunk)

    features_labels = list(zip(features, labels))
    random.shuffle(features_labels)
    features, labels = map(list, zip(*features_labels))
    train_test_border = int((1 - test_size) * len(features))
    features_train, labels_train = __squeeze(features[:train_test_border]), __squeeze(labels[:train_test_border])
    features_test, labels_test = __squeeze(features[train_test_border:]), __squeeze(labels[train_test_border:])
    return np.array(features_train), np.array(labels_train), np.array(features_test), np.array(labels_test)


def tensorify(features, labels):
    return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


def johnson_transform(data, data_len_factor=0.75):
    result = list()
    for i in range(int(len(data) * data_len_factor)):
        result.append(sum(abs(data[j + i] - data[j]) for j in range(len(data) - i)) / (len(data) - i))
    return np.array(result)


def augment_noisy_windows(features, labels, addition_n, noise_variation=1.0):
    random_indices = random.choices(range(0, features.shape[0]), k=addition_n)
    addition_features = np.array([features[i, :, :] + np.random.normal(0, noise_variation, (features.shape[1], features.shape[2]))
                                  for i in random_indices])
    addition_labels = np.array([labels[i] for i in random_indices])
    return np.vstack((features, addition_features)), np.concatenate((labels, addition_labels))


def divide_windows(features, labels, window_size, step_factor):
    step = int(window_size * step_factor)
    new_features, new_labels = list(), list()
    for i in range(0, features.shape[0] - window_size + 1, step):
        if len(set(labels[i:i + window_size])) != 1:
            continue
        new_features.append([features[i:i + window_size, j] for j in range(features.shape[1])])
        new_labels.append(labels[i])
    return np.array(new_features), np.array(new_labels)


def prepare_for_lstm(features, labels):
    features = np.transpose(features, axes=(0, 2, 1))
    return features, labels


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


def normalize_windows(features, labels):
    new_features, new_labels = list(), list()
    for i in range(features.shape[0]):
        row = [features[i, j, :] / max(features[i, j, :]) for j in range(features.shape[1])]
        new_features.append(row)
        new_labels.append(labels[i])
    return np.array(new_features), np.array(new_labels)


def sax_bop_windows(features, labels, n_bins=4):
    new_features, new_labels = list(), list()
    for i in range(features.shape[0]):
        sax = SymbolicAggregateApproximation(n_bins=n_bins, alphabet='ordinal', strategy='uniform')
        pattern_row = sax.fit_transform(features[i, :, :])
        row = [[sum(pattern_row[j, :] == k) for k in range(max(pattern_row[j, :]) + 1)] for j in range(len(pattern_row))]
        print(row)
        new_features.append(row)
        new_labels.append(labels[i])
    return np.array(new_features), np.array(new_labels)


def correlation_windows(features, labels, corr):
    new_features, new_labels = list(), list()
    for i in range(features.shape[0]):
        row = [max(np.correlate(features[i, c[0], features.shape[2] // 4:3 * features.shape[2] // 4], features[i, c[1], j:j + features.shape[2] // 2])
                  for j in range(features.shape[2] // 2)) for c in corr]
        # row = [np.correlate(features[i, c[0], :], features[i, c[1], :]) for c in corr]
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


def transform_data(features, labels, pipeline):
    for stage in pipeline:
        features, labels = stage[0](features, labels, **stage[1])
    return features, labels


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

    X_train, y_train, X_test, y_test = load_data('data/0002_dataset.csv', 0.16666)
    print(len(X_train), len(y_train), len(X_test), len(y_test))
    print(X_test, y_test)

    pass
