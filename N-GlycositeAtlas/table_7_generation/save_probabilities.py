from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import pandas as pd
import numpy as np
import csv
from sklearn.feature_selection import mutual_info_classif


def preprocess_the_dataset(feature_X):

    pt = PowerTransformer()
    pt.fit(feature_X)
    feature_X = pt.transform(feature_X)

    return feature_X


def model_fit(model_name, X_train, y_train):
    if model_name == 'RF':
        model = RandomForestClassifier(random_state=1)
    elif model_name == 'ET':
        model = ExtraTreesClassifier(random_state=1)
    elif model_name == 'DT':
        model = DecisionTreeClassifier(random_state=1)
    elif model_name == 'MLP':
        model = MLPClassifier(random_state=1, max_iter=4000)
    elif model_name == 'LR':
        model = LogisticRegression(class_weight='balanced', random_state=1, max_iter=1000)
    elif model_name == 'SVM':
        model = SVC(kernel='rbf', random_state=1, probability=True)
    elif model_name == 'NB':
        model = GaussianNB()
    elif model_name == 'KNN':
        model = KNeighborsClassifier()
    elif model_name == 'XGB':
        model = XGBClassifier(random_state=1)
    elif model_name == 'PLS':
        model = PLSRegression(n_components=2)
    else:
        print('Wrong model name')
        return

    try:
        model.fit(X_train, y_train)
    except:
        model = PLSRegression(n_components=1)
        model.fit(X_train, y_train)

    return model


def make_string(s):
    str = ''
    for i in s:
        str += i + ", "
    return str[:-2]


feature_paths = {
    'ProtT5-XL-U50-Local': '/media/nafiislam/New Volume/For side thesis/all_features/ProtT5-XL-U50-Local/fasta_train_NGlycositeAtlas.csv',
    'ESM-2-Global': '/media/nafiislam/New Volume/For side thesis/all_features/ESM-2-Global/fasta_train_NGlycositeAtlas.csv',
    'ProteinBert': '/media/nafiislam/New Volume/For side thesis/all_features/ProteinBert/fasta_train_NGlycositeAtlas.csv',
}

learner_combination = [
    ['SVM'], ['XGB'], ['ET'], ['MLP'], ['PLS'], ['LR'], ['NB'], ['RF'], ['DT'], ['KNN'],
    ['SVM', 'XGB'], ['SVM', 'ET'], ['SVM', 'MLP'], ['SVM', 'PLS'], ['SVM', 'LR'], ['SVM', 'NB'], ['SVM', 'RF'], ['SVM', 'DT'], ['SVM', 'KNN'],
    ['SVM', 'XGB', 'ET'], ['SVM', 'XGB', 'MLP'], ['SVM', 'XGB', 'PLS'], ['SVM', 'XGB', 'LR'], ['SVM', 'XGB', 'NB'], ['SVM', 'XGB', 'RF'], ['SVM', 'XGB', 'DT'], ['SVM', 'XGB', 'KNN'],
    ['SVM', 'XGB', 'LR', 'ET'], ['SVM', 'XGB', 'LR', 'MLP'], ['SVM', 'XGB', 'LR', 'PLS'], ['SVM', 'XGB', 'LR', 'NB'], ['SVM', 'XGB', 'LR', 'RF'], ['SVM', 'XGB', 'LR', 'DT'], ['SVM', 'XGB', 'LR', 'KNN'],
    ['SVM', 'XGB', 'LR', 'MLP', 'ET'], ['SVM', 'XGB', 'LR', 'MLP', 'PLS'], ['SVM', 'XGB', 'LR', 'MLP', 'NB'], ['SVM', 'XGB', 'LR', 'MLP', 'RF'], ['SVM', 'XGB', 'LR', 'MLP', 'DT'], ['SVM', 'XGB', 'LR', 'MLP', 'KNN'],
    ['SVM', 'XGB', 'LR', 'MLP', 'KNN', 'ET'], ['SVM', 'XGB', 'LR', 'MLP', 'KNN', 'PLS'], ['SVM', 'XGB', 'LR', 'MLP', 'KNN', 'NB'], ['SVM', 'XGB', 'LR', 'MLP', 'KNN', 'RF'], ['SVM', 'XGB', 'LR', 'MLP', 'KNN', 'DT'],
    ['SVM', 'XGB', 'LR', 'MLP', 'KNN', 'PLS', 'ET'], ['SVM', 'XGB', 'LR', 'MLP', 'KNN', 'PLS', 'NB'], ['SVM', 'XGB', 'LR', 'MLP', 'KNN', 'PLS', 'RF'], ['SVM', 'XGB', 'LR', 'MLP', 'KNN', 'PLS', 'DT'],
    ['SVM', 'XGB', 'LR', 'MLP', 'KNN', 'PLS', 'RF', 'ET'], ['SVM', 'XGB', 'LR', 'MLP', 'KNN', 'PLS', 'RF', 'NB'], ['SVM', 'XGB', 'LR', 'MLP', 'KNN', 'PLS', 'RF', 'DT'],
    ['SVM', 'XGB', 'LR', 'MLP', 'KNN', 'PLS', 'RF', 'ET', 'NB'], ['SVM', 'XGB', 'LR', 'MLP', 'KNN', 'PLS', 'RF', 'ET', 'DT'],
    ['SVM', 'XGB', 'LR', 'MLP', 'KNN', 'PLS', 'RF', 'ET', 'NB', 'DT'],
]

others = ['ESM-2-Global', 'ProtT5-XL-U50-Local']
file_path_Benchmark_embeddings = feature_paths['ProteinBert']
D_feature = pd.read_csv(file_path_Benchmark_embeddings, header=None, low_memory=False)
feature_y_Benchmark_embeddings = D_feature.iloc[:, 0].values

feature_X_Benchmark_embeddings = np.zeros((feature_y_Benchmark_embeddings.shape[0], 1), dtype=float)

file_path_Benchmark_embeddings = feature_paths['ProteinBert']
D_feature = pd.read_csv(file_path_Benchmark_embeddings, header=None, low_memory=False)
feature_X_Benchmark_embeddings = np.concatenate((feature_X_Benchmark_embeddings, D_feature.iloc[:, :].values), axis=1)

feature_X_Benchmark_embeddings = np.delete(feature_X_Benchmark_embeddings, 2, axis=1)
feature_X_Benchmark_embeddings = np.delete(feature_X_Benchmark_embeddings, 0, axis=1)

for other in others:
    file_path_Benchmark_embeddings = feature_paths[other]
    D_feature = pd.read_csv(file_path_Benchmark_embeddings, header=None, low_memory=False)
    feature_X_Benchmark_embeddings = np.concatenate((feature_X_Benchmark_embeddings, D_feature.iloc[:, 2:].values), axis=1)

feature_X_Benchmark_embeddings_positive = feature_X_Benchmark_embeddings[feature_X_Benchmark_embeddings[:, 0] == 1, 1:]
feature_y_Benchmark_embeddings_positive = feature_X_Benchmark_embeddings[feature_X_Benchmark_embeddings[:, 0] == 1, 0].astype('int')

feature_X_Benchmark_embeddings_negative = feature_X_Benchmark_embeddings[feature_X_Benchmark_embeddings[:, 0] == 0, 1:]
feature_y_Benchmark_embeddings_negative = feature_X_Benchmark_embeddings[feature_X_Benchmark_embeddings[:, 0] == 0, 0].astype('int')

print(feature_X_Benchmark_embeddings_positive.shape)
print(feature_y_Benchmark_embeddings_positive.shape)

print(feature_X_Benchmark_embeddings_negative.shape)
print(feature_y_Benchmark_embeddings_negative.shape)

feature_X_Benchmark_embeddings_positive_train, feature_X_Benchmark_embeddings_positive_test, feature_y_Benchmark_embeddings_positive_train, feature_y_Benchmark_embeddings_positive_test = train_test_split(feature_X_Benchmark_embeddings_positive, feature_y_Benchmark_embeddings_positive, test_size=3361, random_state=1)
feature_X_Benchmark_embeddings_negative_train, feature_X_Benchmark_embeddings_negative_test, feature_y_Benchmark_embeddings_negative_train, feature_y_Benchmark_embeddings_negative_test = train_test_split(feature_X_Benchmark_embeddings_negative, feature_y_Benchmark_embeddings_negative, test_size=6343, random_state=1)

print(feature_X_Benchmark_embeddings_positive_train.shape)
print(feature_X_Benchmark_embeddings_positive_test.shape)

print(feature_X_Benchmark_embeddings_negative_train.shape)
print(feature_X_Benchmark_embeddings_negative_test.shape)

feature_X_Benchmark_embeddings_train = np.concatenate((feature_X_Benchmark_embeddings_positive_train, feature_X_Benchmark_embeddings_negative_train), axis=0)
feature_y_Benchmark_embeddings_train = np.concatenate((feature_y_Benchmark_embeddings_positive_train, feature_y_Benchmark_embeddings_negative_train), axis=0)
feature_X_Benchmark_embeddings_test = np.concatenate((feature_X_Benchmark_embeddings_positive_test, feature_X_Benchmark_embeddings_negative_test), axis=0)
feature_y_Benchmark_embeddings_test = np.concatenate((feature_y_Benchmark_embeddings_positive_test, feature_y_Benchmark_embeddings_negative_test), axis=0)

print(feature_X_Benchmark_embeddings_train.shape)
print(feature_y_Benchmark_embeddings_train.shape)

print(feature_X_Benchmark_embeddings_test.shape)
print(feature_y_Benchmark_embeddings_test.shape)

feature_X_Benchmark_embeddings_train = preprocess_the_dataset(feature_X_Benchmark_embeddings_train)
feature_X_Benchmark_embeddings_test = preprocess_the_dataset(feature_X_Benchmark_embeddings_test)

X = feature_X_Benchmark_embeddings_train.copy()
y = feature_y_Benchmark_embeddings_train.copy()

rus = RandomUnderSampler(random_state=1)
X, y = rus.fit_resample(X, y)

c = Counter(y)
print(c)

probabilities = {}
model_names = ['SVM', 'XGB', 'ET', 'MLP', 'PLS', 'LR', 'NB', 'RF', 'DT', 'KNN']
for model_name in model_names:
    print(model_name+" model fit is running")
    if model_name == 'PLS':
        model = model_fit(model_name, X, y)
        p_all = []
        p_all.append([1 - np.abs(item[0]) for item in model.predict(feature_X_Benchmark_embeddings_test)])
        p_all.append([np.abs(item[0]) for item in model.predict(feature_X_Benchmark_embeddings_test)])
        probabilities[model_name] = np.transpose(np.array(p_all))[:, 1]
    else:
        model = model_fit(model_name, X, y)
        probabilities[model_name] = model.predict_proba(feature_X_Benchmark_embeddings_test)[:, 1]

    with open('probabilities/'+model_name+'.csv', 'w', newline='') as file:
        for j in probabilities[model_name]:
            file.write(str(j) + ',')
        file.write('\n')

