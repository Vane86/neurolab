import csv
from scipy import fft
from matplotlib import pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from itertools import product

from data_utils import *
from models import *


if __name__ == '__main__':

    # left_channels = [2, 3, 4, 7, 8, 5, 9, 6, 10, 11, 13, 12, 14, 15]
    left_channels = [0, 1, 2, 3, 4, 5]
    # right_channels = [31, 29, 30, 28, 26, 25, 27, 24, 22, 23, 21, 20, 18, 19]
    right_channels = [6, 7, 8, 9, 10, 11, 12]
    data_pipeline = [
        (divide_windows, {'window_size': 64, 'step_factor': 0.0625}),
        (augment_noisy_windows, {'addition_n': 1024, 'noise_variation': 2.0}),
        (fourier_windows, {}),
        # (prepare_for_lstm, {}),
        # (plot_windows, {}),
        # (correlation_windows, {'corr': [(2, 31), (3, 30), (4, 29), (8, 25),
        #                                 (7, 26), (5, 28), (9, 24), (6, 27),
        #                                 (10, 23), (11, 22), (13, 20), (12, 21),
        #                                 (14, 19), (15, 18)]}),
        # (correlation_windows, {'corr': list(product(left_channels, right_channels))}),
        # (sax_bop_windows, {}),
        # (normalize_windows, {}),
        (concat_by_channels, {}),
        (tensorify, {})
    ]

    features_train, labels_train, features_test, labels_test = load_data('data/face_dataset.csv', test_size=0.25)
    features_train, labels_train = transform_data(features_train, labels_train, data_pipeline)
    features_test, labels_test = transform_data(features_test, labels_test, data_pipeline[0:1] + data_pipeline[2:])

    print('Train shape:', features_train.shape)
    print('Test shape:', features_test.shape)

    # clf = MLPClassifier(hidden_layer_sizes=(64,))
    # clf = GradientBoostingClassifier(n_estimators=100)

    clf = Classifier(FCNN(features_train.shape[1], len(set(labels_train)), hidden_size=16, dropout_p=0.1),
                     epochs=15,
                     learning_rate=0.001,
                     lamb=0.0,
                     batch_size=256)
    # clf = Classifier(LSTMNN(features_train.shape[2], len(set(labels_train)), hidden_size=8, num_layers=1),
    #                  epochs=500,
    #                  learning_rate=0.001,
    #                  lamb=2,
    #                  batch_size=64)
    clf.fit(features_train, labels_train)

    print('Train score:', f1_score(clf.predict(features_train), labels_train, average='weighted'))
    print('Test acc:', accuracy_score(clf.predict(features_train), labels_train))
    print('Test score:', f1_score(clf.predict(features_test), labels_test, average='weighted'))
    print('Test acc:', accuracy_score(clf.predict(features_test), labels_test))

    # Error analysis:
    errors = dict()
    for index in range(features_test.shape[0]):
        key = (labels_test[index].item(), clf.predict(features_test[index:index + 1]).item())
        errors.setdefault(key, 0)
        errors[key] += 1
    print(errors)
