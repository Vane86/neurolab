import csv
from scipy import fft
from matplotlib import pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler

from data_utils import *
from models import *


if __name__ == '__main__':

    K_features = 750
    nn_classifier = Classifier(FCNN(K_features, 5), 3000, learning_rate=0.001, lamb=0.01)
    pipeline(nn_classifier, [(divide_windows, {'window_size': 128, 'step_factor': 0.5}),
                             # (fourier_windows, {}),
                             (johnson_windows, {}),
                             (differential_windows, {}),
                             # (avg_windows, {'n_windows': 2, 'step_windows': 1}),
                             (concat_by_channels, {})],
             'data/data_fists_knees_train.csv', 'data/data_fists_knees_test.csv',
             data_type='tensor',
             label2class=['neutral', 'left_fist',
                          'left_knee', 'right_fist',
                          'right_knee'],
             features_processors=[StandardScaler()],
             # features_selectors=[SelectKBest(mutual_info_classif, k=K_features)]
             )

    # raw_features, raw_labels = load_data('data/data_fists_knees.csv', label2class=['neutral', 'left_fist',
    #                                                                                'left_knee', 'right_fist',
    #                                                                                'right_knee'])
    # # raw_features, raw_labels = load_data('data/data_fists_knees_train.csv')
    #
    # # features, labels = divide_windows(raw_features, raw_labels, window_size=128, step_factor=0.5)
    # # features, labels = fourier_windows(features, labels)
    # # features, labels = avg_windows(features, labels, n_windows=3, step_windows=1)
    # # plot_windows(features, labels, only=(1, -2, 2, -3, 3, -4))
    #
    # features, labels = divide_windows(raw_features, raw_labels, window_size=128, step_factor=0.5)
    # features, labels = johnson_windows(features, labels)
    # features, labels = differential_windows(features, labels)
    # features, labels = avg_windows(features, labels, n_windows=2, step_windows=1)
    # # plot_windows(features, labels, only=(0, 1, 2))
    #
    # features, labels = concat_by_channels(features, labels)
    # features_train, features_val, labels_train, labels_val = train_test_split(features, labels, test_size=0.25)
    # # print(labels)
    #
    # # scaler = StandardScaler()
    # # scaler.fit(features)
    # # features_train = scaler.transform(features_train)
    # # features_val = scaler.transform(features_val)
    #
    # # pca = PCA(n_components=features_train.shape[1]).fit(features_train)
    # # features_train = pca.transform(features_train)
    # # features_test = pca.transform(features_test)
    # # print(sum(pca.explained_variance_ratio_))
    #
    # # print(features_train.shape, len(features_val))
    # # for i in range(10):
    # #     plt.plot(johnson_transform(raw_features[:128, i]))
    # # plt.legend(['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4'])
    # # plt.show()
    #
    # # clf = MLPClassifier(max_iter=300, hidden_layer_sizes=(10,)).fit(features_train, labels_train)
    # # clf = BaggingClassifier(base_estimator=MLPClassifier(max_iter=1000), n_estimators=15)
    # # clf = GradientBoostingClassifier(n_estimators=500)
    #
    # features_train, labels_train = torch.tensor(features_train, dtype=torch.float32), torch.tensor(labels_train, dtype=torch.long)
    # features_val, labels_val = torch.tensor(features_val, dtype=torch.float32), torch.tensor(labels_val, dtype=torch.long)
    # clf = Classifier(FCNN(features_train.shape[1], 5), epochs=1000, learning_rate=0.003, lamb=0.05)
    #
    # val_acc_history = clf.fit(features_train, labels_train, features_val, labels_val)
    #
    # print('Train score:', f1_score(clf.predict(features_train), labels_train, average='weighted'))
    # print('Validation score:', f1_score(clf.predict(features_val), labels_val, average='weighted'))
    #
    # features_test, labels_test = load_data('data/data_fists_knees.csv', label2class=['neutral', 'left_fist',
    #                                                                                  'left_knee', 'right_fist',
    #                                                                                  'right_knee'])
    #
    # print(labels_test)
    #
    # features_test, labels_test = divide_windows(features_test, labels_test, window_size=128, step_factor=0.7)
    # features_test, labels_test = johnson_windows(features_test, labels_test)
    # features_test, labels_test = differential_windows(features_test, labels_test)
    # features_test, labels_test = avg_windows(features_test, labels_test, n_windows=2, step_windows=1)
    # # plot_windows(features, labels, only=(0, 1, 2, 3, 4))
    #
    # features_test, labels_test = concat_by_channels(features_test, labels_test)
    #
    # features_test, labels_test = torch.tensor(features_test, dtype=torch.float32), torch.tensor(labels_test, dtype=torch.long)
    # print('Test score:', f1_score(clf.predict(features_test), labels_test, average='weighted'))
    # # # plot_windows(features_test, labels_test, only=(0, -1))
    # #
    # # features_test, labels_test = avg_windows(features_test, labels_test, n_windows=3, step_windows=1)
    # # features_test, labels_test = concat_by_channels(features_test, labels_test)
    # #
    # # features_test = scaler.transform(features_test)
    # # print('Test score:', f1_score(clf.predict(features_test), labels_test, average='weighted'))
    # #
    # # test_predicts = clf.predict(features_test)
    # # # Correction:
    # # # new_test_predict = list(test_predicts[:4])
    # # # for i in range(4, len(test_predicts)):
    # # #     new_test_predict.append(max([(x, list(test_predicts[i - 4:i]).count(x)) for x in test_predicts[i - 4:i]], key=lambda x: x[1])[0])
    # # plt.plot(range(len(test_predicts)), test_predicts)
    # # plt.plot(range(len(test_predicts)), labels_test)
    # # plt.show()
    # #
    # # errors = [0] * len(set(labels))
    # # for i, label in enumerate(labels_test):
    # #     if label != test_predicts[i]:
    # #         errors[label] += 1
    # # errors = [errors[i] / len(labels_test[labels_test == i]) for i in range(len(errors))]
    # # print(errors)
