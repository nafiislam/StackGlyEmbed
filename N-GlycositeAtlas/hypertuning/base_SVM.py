from sklearn.metrics import accuracy_score,confusion_matrix, roc_auc_score, f1_score, matthews_corrcoef, average_precision_score
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import pandas as pd
import numpy as np
import csv
import random


def preprocess_the_dataset(feature_X):

    pt = PowerTransformer()
    pt.fit(feature_X)
    feature_X = pt.transform(feature_X)

    return feature_X


def find_metrics(model_name, y_test, C, kernel, gamma, degree):
    if model_name == 'SVM':
        model = SVC(random_state=1, probability=True, kernel=kernel, C=C, gamma=gamma, degree=degree)
    else:
        print('Wrong model name')
        return

    model.fit(X_train, y_train)

    y_predict = model.predict(X_test)  # predicted labels
    y_proba = model.predict_proba(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, y_predict).ravel()  # y_true, y_pred

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    bal_acc = balanced_accuracy_score(y_test, y_predict)
    acc = accuracy_score(y_test, y_predict)

    if tp == 0 and fp == 0:
        prec = 0
    else:
        prec = tp / (tp + fp)

    if prec == 0 and sensitivity == 0:
        f1_score_1 = 0
    else:
        f1_score_1 = 2 * prec * sensitivity / (prec + sensitivity)
    mcc = matthews_corrcoef(y_test, y_predict)
    auc = roc_auc_score(y_test, y_proba[:, 1])
    auPR = average_precision_score(y_test, y_proba[:, 1])  # auPR

    return sensitivity, specificity, bal_acc, acc, prec, f1_score_1, mcc, auc, auPR


feature_paths = {
    'ProtT5-XL-U50-Local': '/media/nafiislam/New Volume/For side thesis/all_features/ProtT5-XL-U50-Local/fasta_train_NGlycositeAtlas.csv',
    'ESM-2-Global': '/media/nafiislam/New Volume/For side thesis/all_features/ESM-2-Global/fasta_train_NGlycositeAtlas.csv',
    'ProteinBert': '/media/nafiislam/New Volume/For side thesis/all_features/ProteinBert/fasta_train_NGlycositeAtlas.csv',
}

others = ['ESM-2-Global', 'ProtT5-XL-U50-Local']

file_path_Benchmark_embeddings = feature_paths['ProteinBert']
D_feature = pd.read_csv(file_path_Benchmark_embeddings, header=None, low_memory=False)
feature_y_Benchmark_embeddings = D_feature.iloc[:, 0].values

feature_X_Benchmark_embeddings = np.zeros((feature_y_Benchmark_embeddings.shape[0], 1), dtype=float)

file_path_Benchmark_embeddings = feature_paths['ProteinBert']
D_feature = pd.read_csv(file_path_Benchmark_embeddings, header=None, low_memory=False)
feature_X_Benchmark_embeddings = np.concatenate((feature_X_Benchmark_embeddings, D_feature.iloc[:, :].values),
                                                axis=1)

feature_X_Benchmark_embeddings = np.delete(feature_X_Benchmark_embeddings, 2, axis=1)
feature_X_Benchmark_embeddings = np.delete(feature_X_Benchmark_embeddings, 0, axis=1)

for other in others:
    file_path_Benchmark_embeddings = feature_paths[other]
    D_feature = pd.read_csv(file_path_Benchmark_embeddings, header=None, low_memory=False)
    feature_X_Benchmark_embeddings = np.concatenate((feature_X_Benchmark_embeddings, D_feature.iloc[:, 2:].values),
                                                    axis=1)

feature_X_Benchmark_embeddings_positive = feature_X_Benchmark_embeddings[feature_X_Benchmark_embeddings[:, 0] == 1,
                                          1:]
feature_y_Benchmark_embeddings_positive = feature_X_Benchmark_embeddings[
    feature_X_Benchmark_embeddings[:, 0] == 1, 0].astype('int')

feature_X_Benchmark_embeddings_negative = feature_X_Benchmark_embeddings[feature_X_Benchmark_embeddings[:, 0] == 0,
                                          1:]
feature_y_Benchmark_embeddings_negative = feature_X_Benchmark_embeddings[
    feature_X_Benchmark_embeddings[:, 0] == 0, 0].astype('int')

print(feature_X_Benchmark_embeddings_positive.shape)
print(feature_y_Benchmark_embeddings_positive.shape)

print(feature_X_Benchmark_embeddings_negative.shape)
print(feature_y_Benchmark_embeddings_negative.shape)

feature_X_Benchmark_embeddings_positive_train, feature_X_Benchmark_embeddings_positive_test, feature_y_Benchmark_embeddings_positive_train, feature_y_Benchmark_embeddings_positive_test = train_test_split(feature_X_Benchmark_embeddings_positive, feature_y_Benchmark_embeddings_positive, test_size=3361, random_state=1)
feature_X_Benchmark_embeddings_negative_train, feature_X_Benchmark_embeddings_negative_test, feature_y_Benchmark_embeddings_negative_train, feature_y_Benchmark_embeddings_negative_test = train_test_split(feature_X_Benchmark_embeddings_negative, feature_y_Benchmark_embeddings_negative, test_size=6343, random_state=1)

feature_X_Benchmark_embeddings_train = np.concatenate(
    (feature_X_Benchmark_embeddings_positive_train, feature_X_Benchmark_embeddings_negative_train), axis=0)
feature_y_Benchmark_embeddings_train = np.concatenate(
    (feature_y_Benchmark_embeddings_positive_train, feature_y_Benchmark_embeddings_negative_train), axis=0)
feature_X_Benchmark_embeddings_test = np.concatenate(
    (feature_X_Benchmark_embeddings_positive_test, feature_X_Benchmark_embeddings_negative_test), axis=0)
feature_y_Benchmark_embeddings_test = np.concatenate(
    (feature_y_Benchmark_embeddings_positive_test, feature_y_Benchmark_embeddings_negative_test), axis=0)

print(feature_X_Benchmark_embeddings_train.shape)
print(feature_y_Benchmark_embeddings_train.shape)

X = feature_X_Benchmark_embeddings_train.copy()
y = feature_y_Benchmark_embeddings_train.copy()

X = preprocess_the_dataset(X)

# balance the dataset :
rus = RandomUnderSampler(random_state=1)
X, y = rus.fit_resample(X, y)

c = Counter(y)
print(c)

#gamma='scale', kernel='rbf', C=1, degree=3
grid = {
    'C': [0.01, 0.1, 1, 10],
    'kernel': ["linear", "poly", "rbf", "sigmoid"],
    'gamma': ['scale', 'auto'],
    'degree': [3, 5],
}

with open("./results_base_SVM.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["C, kernel, gamma, degree", "Sensitivity", "Specificity", "Balanced_acc", "Accuracy", "Precision", "F1-score", "MCC", "AUC", "auPR"])
    for C in grid['C']:
        for kernel in grid['kernel']:
            for gamma in grid['gamma']:
                for degree in grid['degree']:
                    random.seed(1)

                    # Step 06 : Spliting with 5-FCV :
                    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

                    local_Sensitivity = []
                    local_Specificity = []
                    local_Balanced_acc = []
                    local_Accuracy = []
                    local_Precision = []
                    local_AUPR = []
                    local_F1 = []
                    local_MCC = []
                    local_AUC = []

                    i = 1
                    for train_index, test_index in cv.split(X, y):
                        X_train = X[train_index]
                        X_test = X[test_index]

                        y_train = y[train_index]
                        y_test = y[test_index]

                        sensitivity, specificity, bal_acc, acc, prec, f1_score_1, mcc, auc, auPR = find_metrics('SVM', y_test, C, kernel, gamma, degree)

                        local_Sensitivity.append(sensitivity)
                        local_Specificity.append(specificity)
                        local_Balanced_acc.append(bal_acc)
                        local_Accuracy.append(acc)
                        local_Precision.append(prec)
                        local_F1.append(f1_score_1)
                        local_MCC.append(mcc)
                        local_AUC.append(auc)
                        local_AUPR.append(auPR)

                        print(i, 'th iteration done')
                        i = i + 1
                        print(
                            '___________________________________________________________________________________________________________')

                    print(C, kernel, gamma, degree)
                    print('Sensitivity : {0:.3f}'.format(np.mean(local_Sensitivity)))
                    print('Specificity : {0:.3f}'.format(np.mean(local_Specificity)))
                    print('Balanced_acc : {0:.3f}'.format(np.mean(local_Balanced_acc)))
                    print('Accuracy : {0:.3f}'.format(np.mean(local_Accuracy)))
                    print('Precision : {0:.3f}'.format(np.mean(local_Precision)))
                    print('F1-score: {0:.3f}'.format(np.mean(local_F1)))
                    print('MCC: {0:.3f}'.format(np.mean(local_MCC)))
                    print('AUC: {0:.3f}'.format(np.mean(local_AUC)))
                    print('auPR: {0:.3f}'.format(np.mean(local_AUPR)))

                    writer.writerow([f"'C':{C}, 'kernel':{kernel}, 'gamma':{gamma}, 'degree': {degree}", '{0:.3f}'.format(np.mean(local_Sensitivity)),
                                     '{0:.3f}'.format(np.mean(local_Specificity)),
                                     '{0:.3f}'.format(np.mean(local_Balanced_acc)),
                                     '{0:.3f}'.format(np.mean(local_Accuracy)),
                                     '{0:.3f}'.format(np.mean(local_Precision)), '{0:.3f}'.format(np.mean(local_F1)),
                                     '{0:.3f}'.format(np.mean(local_MCC)), '{0:.3f}'.format(np.mean(local_AUC)),
                                     '{0:.3f}'.format(np.mean(local_AUPR))])
                    print('___________________________________________________________________________________________________________')