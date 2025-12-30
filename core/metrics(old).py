import numpy as np
import pandas as pd
from sklearn import metrics


def purity_score(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]

    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels) + 1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return metrics.accuracy_score(y_true, y_voted_labels)


def calculation(index, ground_truth, predict):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(0, len(index) - 1):
        report1_gt = ground_truth[i]
        report1_pre = predict[i]
        report2_gt = ground_truth[i + 1]
        report2_pre = predict[i + 1]
        if report1_gt == report2_gt:
            if report1_pre == report2_pre:
                TP = TP + 1
            else:
                FN = FN + 1
        else:
            if report1_pre == report2_pre:
                FP = FP + 1
            else:
                TN = TN + 1

    RI = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return RI, precision, recall


def metric(label, cluster_k, type_dict):
    df = pd.read_csv(label, header=None, encoding='utf-8')
    index = []
    ground_truth = []
    predict = []

    start_idx = 0
    # Check if first row is header
    if len(df) > 0 and str(df.loc[0][0]) == 'index':
        start_idx = 1

    for i in range(start_idx, len(df)):
        index.append(df.loc[i][0])
        ground_truth.append(df.loc[i][5])
        predict.append(df.loc[i][4])

    for i in range(1, cluster_k + 1):
        index_list = type_dict[i]
        for k in index_list:
            item = index.index(k)
            predict[item - 1] = i

    ARI = metrics.adjusted_rand_score(ground_truth, predict)
    NMI = metrics.normalized_mutual_info_score(ground_truth, predict)

    ground_truth_np = np.array(ground_truth)
    predict_np = np.array(predict)
    purity = purity_score(ground_truth_np, predict_np)

    RI, precision, recall = calculation(index, ground_truth, predict)

    return ARI, NMI, purity, RI, precision, recall
