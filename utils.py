import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def set_seed(seed):
    """
    设置随机种子，确保实验可复现
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def target_distribution(batch):
    """
    计算目标分布 p_ij，用于 KL 散度损失：
    参考 Xie/Girshick/Farhadi (2016) 的 DEC 方法

    :param batch: [batch_size, num_clusters] 软分配概率矩阵 P
    :return: [batch_size, num_clusters] 目标分布
    """
    weight = (batch ** 2) / torch.sum(batch, 0)  # 计算权重
    return (weight.t() / torch.sum(weight, 1)).t()  # 归一化


def inv_lr_scheduler(optimizer, lr, iter, max_iter, gamma=10, power=0.75):
    """
    逆学习率调度（Inverse Learning Rate Scheduler）
    :param optimizer: 优化器
    :param lr: 初始学习率
    :param iter: 当前迭代次数
    :param max_iter: 总迭代次数
    :param gamma: 调节因子，默认 10
    :param power: 下降速率，默认 0.75
    """
    learning_rate = lr * (1 + gamma * (float(iter) / float(max_iter))) ** (-power)
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate * param_group["lr_mult"]
    return optimizer


def KL_divergence(output, target):
    """
    计算 KL 散度，用于聚类损失
    :param output: [batch_size, num_clusters] 预测的概率分布 P
    :param target: [batch_size, num_clusters] 目标分布 P_target
    :return: KL 散度损失
    """
    return F.kl_div(output.log(), target, reduction="batchmean")


def JS_divergence(output, target):
    """
    计算 Jensen-Shannon 散度，用于衡量两个分布的相似性
    :param output: [batch_size, num_clusters] 预测分布 P
    :param target: [batch_size, num_clusters] 目标分布 P_target
    :return: JS 散度
    """
    KLD = nn.KLDivLoss(reduction='batchmean')
    M = 0.5 * (output + target)  # 计算中间分布 M
    return 0.5 * KLD(output.log(), M) + 0.5 * KLD(target.log(), M)


def CS_divergence(output, target):
    """
    计算 Cauchy-Schwarz 散度
    :param output: [batch_size, num_clusters] 预测分布 P
    :param target: [batch_size, num_clusters] 目标分布 P_target
    :return: CS 散度
    """
    numerator = torch.sum(output * target)
    denominator = torch.sqrt(torch.sum(output**2) * torch.sum(target**2))
    return -torch.log(numerator / denominator)


class AverageMeter:
    """
    计算并存储均值和当前值（用于训练监控）
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
