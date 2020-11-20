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


from data_utils import *

if __name__ == '__main__':

    # raw_features, raw_labels = load_data('data/data_fists_knees.csv')
    raw_features, raw_labels = load_data('data/data_fists_knees_train.csv')

    features, labels = divide_windows(raw_features, raw_labels, window_size=128, step_factor=0.25)
    # features, labels = avg_windows(features, labels, n_windows=3, step_windows=3)
    features, labels = fourier_windows(features, labels)

    plot_windows(features, labels, only=(1, -2, 2, -3, 3, -4))

    features, labels = concat_by_channels(features, labels)
    features_train, features_val, labels_train, labels_val = train_test_split(features, labels, test_size=0.15)

    scaler = StandardScaler()
    scaler.fit(features)
    features_train = scaler.transform(features_train)
    features_val = scaler.transform(features_val)

    # pca = PCA(n_components=features_train.shape[1]).fit(features_train)
    # features_train = pca.transform(features_train)
    # features_test = pca.transform(features_test)
    # print(sum(pca.explained_variance_ratio_))

    print(features_train.shape, len(features_val))

    clf = MLPClassifier(max_iter=300, hidden_layer_sizes=(100,)).fit(features_train, labels_train)
    # clf = BaggingClassifier(base_estimator=MLPClassifier(max_iter=1000), n_estimators=15)
    # clf = GradientBoostingClassifier(n_estimators=500)

    clf.fit(features_train, labels_train)

    print('Train score:', f1_score(clf.predict(features_train), labels_train, average='weighted'))
    print('Validation score:', f1_score(clf.predict(features_val), labels_val, average='weighted'))

    features_test, labels_test = load_data('data/data_fists_knees_test.csv')

    features_test, labels_test = divide_windows(features_test, labels_test, window_size=128, step_factor=0.25)
    # features, labels = avg_windows(features, labels, n_windows=3, step_windows=3)
    features_test, labels_test = fourier_windows(features_test, labels_test)
    features_test, labels_test = concat_by_channels(features_test, labels_test)

    features_test = scaler.transform(features_test)
    print('Test score:', f1_score(clf.predict(features_test), labels_test, average='weighted'))

    test_predicts = clf.predict(features_test)
    # Correction:
    # new_test_predict = list(test_predicts[:4])
    # for i in range(4, len(test_predicts)):
    #     new_test_predict.append(max([(x, list(test_predicts[i - 4:i]).count(x)) for x in test_predicts[i - 4:i]], key=lambda x: x[1])[0])
    # plt.plot(range(len(test_predicts)), new_test_predict)
    # plt.plot(range(len(test_predicts)), labels_test)
    # plt.show()

    errors = [0] * len(set(labels))
    for i, label in enumerate(labels_test):
        if label != test_predicts[i]:
            errors[label] += 1
    errors = [errors[i] / len(labels_test[labels_test == i]) for i in range(len(errors))]
    print(errors)

    # concat_csvs(('data/back.csv', 'data/forward.csv', 'data/left.csv', 'data/neutral.csv', 'data/right.csv'), 'data/data.csv')
