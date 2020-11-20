import csv

import numpy as np
from scipy import fft

from matplotlib import pyplot as plt


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

# def extract_features_fourier(raw_features, labels, window_size=128, step_factor=0.5):
#     step = int(window_size * step_factor)
#     new_features, new_labels = list(), list()
#     for i in range(0, raw_features.shape[0] - window_size, step):
#         if len(set(labels[i:i + window_size])) != 1:
#             continue
#         channel_ffts = list()
#         for j in range(raw_features.shape[1]):
#             window = raw_features[i:i + window_size, j]
#             win_mean = window.mean()
#             channel_ffts.extend([np.log1p(abs(x)) for x in fft(window - win_mean)[:window_size // 2]])
#         new_features.append(channel_ffts)
#         new_labels.append(labels[i])
#     return np.array(new_features), np.array(new_labels)


if __name__ == '__main__':
    test_features = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    test_labels = [1, 1, 2, 2]
    windows, labels = divide_windows(test_features, test_labels, 2, 1.0)
    print(windows, labels)
    print()
    avg, labels = avg_windows(windows, [1, 1], 2, 1)
    print(avg, labels)
    print()
    fourier, labels = fourier_windows(avg, labels)
    print(fourier, labels)
    print()
    channels, labels = concat_by_channels(windows, [1, 2])
    print(channels, labels)

    sine = np.array([np.sin(10 * np.linspace(-10, 10, 500))]).T
    f, l = divide_windows(sine, [1] * len(sine), window_size=50, step_factor=1.0)
    f, l = fourier_windows(f, l)
    plot_windows(f[:, 0, :])

    noisy_sine = np.array([np.sin(10 * np.linspace(-1000, 1000, 10000)) + np.random.normal(0, 1, 10000)]).T
    f, l = divide_windows(noisy_sine, [1] * len(noisy_sine), window_size=50, step_factor=1.0)
    f_avg, l_avg = avg_windows(f, l, n_windows=20, step_windows=20)
    f_f, l_f = fourier_windows(f, l)
    f_af, l_af = fourier_windows(f_avg, l_avg)
    plot_windows(f_f[:, 0, :])
    plot_windows(f_af[:, 0, :])
