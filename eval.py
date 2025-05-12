import numpy as np
import metrics
from sklearn import metrics
from scipy.optimize import linear_sum_assignment

def purity_score(cluster, label):
    cluster = np.array(cluster)
    label = np. array(label)
    indedata1 = {}
    for p in np.unique(label):
        indedata1[p] = np.argwhere(label == p)
    indedata2 = {}
    for q in np.unique(cluster):
        indedata2[q] = np.argwhere(cluster == q)

    count_all = []
    for i in indedata1.values():
        count = []
        for j in indedata2.values():
            a = np.intersect1d(i, j).shape[0]
            count.append(a)
        count_all.append(count)
    return sum(np.max(count_all, axis=0))/len(cluster)

def f_measure(labels_true, labels_pred, beta=1.):
    (tn, fp), (fn, tp) = metrics.cluster.pair_confusion_matrix(labels_true, labels_pred)
    #print(tn,fp,fn,tp)
    #ri = (tp + tn) / (tp + tn + fp + fn)
    #ari = 2. * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    p, r = tp / (tp + fp), tp / (tp + fn)
    f_beta = (1 + beta**2) * (p * r / ((beta ** 2) * p + r))
    return  f_beta
def ACC(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true=np.array(y_true)
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    D = int(D)
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):

        w[int(y_pred[i]), y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) * 1.0 / y_pred.size

import numpy as np

def balance_score(y_pred, sensitive_attr):
    unique_clusters = np.unique(y_pred)
    unique_groups = np.unique(sensitive_attr)
    cluster_sizes = {cluster: np.sum(y_pred == cluster) for cluster in unique_clusters}
    min_proportions = []
    zero_clusters = []

    for cluster in unique_clusters:
        cluster_indices = np.where(y_pred == cluster)[0]
        protected_values = sensitive_attr[cluster_indices]
        group_counts = {group: np.sum(protected_values == group) for group in unique_groups}
        min_proportion = min(group_counts[group] / cluster_sizes[cluster] for group in unique_groups)
        min_proportions.append(min_proportion)

        if min_proportion == 0:
            zero_clusters.append(cluster)



    return np.min(min_proportions)


def fairness_evaluation(y_pred, sensitive_attr):
    """
    计算公平性指标（衡量聚类对受保护属性 G 的影响）

    :param y_pred: 预测的聚类标签 (N,)
    :param sensitive_attr: 受保护属性 G (N,)
    :return: 公平性得分（越接近 1 表示越公平）
    """
    unique_clusters = np.unique(y_pred)  # 获取聚类簇
    fairness_scores = []

    for cluster in unique_clusters:
        cluster_indices = np.where(y_pred == cluster)[0]  # 获取该簇的样本索引
        protected_values = sensitive_attr[cluster_indices]  # 提取该簇的受保护属性

        # 计算受保护属性在该簇的分布
        unique_groups, counts = np.unique(protected_values, return_counts=True)
        proportions = counts / counts.sum()

        # 计算熵（Entropy），衡量受保护属性在该簇的均匀分布程度
        entropy = -np.sum(proportions * np.log2(proportions + 1e-9))  # 避免 log(0)
        max_entropy = np.log2(len(unique_groups)) if len(unique_groups) > 1 else 1  # 理论最大熵

        fairness_scores.append(entropy / max_entropy)  # 归一化到 [0,1]

    return np.mean(fairness_scores)  # 取所有簇的公平性均值