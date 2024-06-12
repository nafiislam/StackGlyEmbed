from sklearn.metrics import accuracy_score, precision_score, recall_score, matthews_corrcoef,roc_auc_score
import os
import random
import pickle as pk
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
import sys
import pickle 
import warnings
warnings.filterwarnings("ignore")


def model(test_x):
    with open('./model/EMNgly.pickle','rb') as f:  
        model = pickle.load(f)
    predict_y_2 = model.predict_proba(test_x)

    predict_y = predict_y_2[:,-1]
    return predict_y.reshape(1,len(test_x))


def get_scores(label_y, predict_y, th=0.5):
    ret_dict = {}
    auc = roc_auc_score(label_y, predict_y)
    ret_dict['auc'] = round(auc,5)
    predict_label = []
    for item in predict_y:
        if item > th:
            predict_label.append(1)
        else:
            predict_label.append(0)

    mcc = matthews_corrcoef(label_y, predict_label)
    ret_dict['mcc'] = round(mcc,5)

    predict_label = np.array(predict_label)
    acc = accuracy_score(label_y, predict_label)
    ret_dict['acc'] = round(acc,5)

    c = predict_label + label_y
    TN = np.where(c==0)[0].shape[0]
    
    specificity = TN/np.where(label_y==0)[0].shape[0]
    ret_dict['specificity'] = round(specificity,5)

    recall = recall_score(label_y, predict_label)
    ret_dict['sensitivity'] = round(recall,5)

    precision = precision_score(label_y, predict_label)
    ret_dict['precision'] = round(precision,5)

    return ret_dict

test_save_file = './data/N-GlycositeAtlas-test.csv'

if __name__ == '__main__':
    print("start reading train and test data...")
    
    test = pd.read_csv(test_save_file)
    test_Y = test.iloc[:,-1:].values.tolist()
    test_Y = np.array(test_Y).reshape(1,len(test_Y))[0]
    test_X = test.iloc[:,1:-1].values.tolist()
    print(test_Y)

    print("test data count:", len(test_X))
    predict_y = model(test_X)
    predict_y = predict_y.tolist()
    scores = get_scores(np.array(test_Y), predict_y[0])
    
    print(scores)
    