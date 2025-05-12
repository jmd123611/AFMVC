import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)

def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()
    return fun1

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class MultiViewEncoder(nn.Module):
    def __init__(self, input_dims, hidden_dim):
        super(MultiViewEncoder, self).__init__()
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU()
            ) for input_dim in input_dims
        ])

    def forward(self, X):
        return [encoder(x) for encoder, x in zip(self.encoders, X)]

class MultiViewDecoder(nn.Module):
    def __init__(self, input_dims, hidden_dim):
        super(MultiViewDecoder, self).__init__()
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, input_dim)
            ) for input_dim in input_dims
        ])

    def forward(self, Z):
        return [decoder(z) for decoder, z in zip(self.decoders, Z)]

class MultiViewAutoEncoder(nn.Module):
    def __init__(self, input_dims, hidden_dim):
        super(MultiViewAutoEncoder, self).__init__()
        self.encoder = MultiViewEncoder(input_dims, hidden_dim)
        self.decoder = MultiViewDecoder(input_dims, hidden_dim)

    def forward(self, X):
        Z = self.encoder(X)
        X_recon = self.decoder(Z)
        return Z, X_recon

class ClusterAssignment(nn.Module):
    def __init__(self, num_clusters, input_dims, hidden_dim, alpha=1.0):
        super(ClusterAssignment, self).__init__()
        self.num_clusters = num_clusters
        self.alpha = alpha
        self.total_hidden_dim = hidden_dim * len(input_dims)

        # ✅ 融合特征的聚类中心
        self.cluster_centers_fused = nn.Parameter(torch.randn(num_clusters, self.total_hidden_dim))

        # ✅ 各视图的单独聚类中心
        self.cluster_centers_views = nn.ParameterList([
            nn.Parameter(torch.randn(num_clusters, hidden_dim)) for _ in input_dims
        ])

    def forward(self, Z_fused, Z_views):
        """
        计算 `P_fused` 和 `P_views`
        """
        # ✅ 计算 `P_fused`
        dist_fused = torch.sum((Z_fused.unsqueeze(1) - self.cluster_centers_fused) ** 2, dim=2)
        q_fused = 1.0 / (1.0 + (dist_fused / self.alpha))
        q_fused **= (self.alpha + 1.0) / 2.0
        q_fused /= torch.sum(q_fused, dim=1, keepdim=True)

        # ✅ 计算 `P_views`
        P_views = []
        for i, Z_v in enumerate(Z_views):
            dist_view = torch.sum((Z_v.unsqueeze(1) - self.cluster_centers_views[i]) ** 2, dim=2)
            q_view = 1.0 / (1.0 + (dist_view / self.alpha))
            q_view **= (self.alpha + 1.0) / 2.0
            q_view /= torch.sum(q_view, dim=1, keepdim=True)
            P_views.append(q_view)

        return q_fused, P_views




class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature, hidden_size, num_classes, max_iter, lr_mult):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        self.ad_layer2 = nn.Linear(hidden_size, num_classes)  # 多分类输出
        self.apply(init_weights)

        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = float(max_iter)
        self.lr_mult = lr_mult

    def forward(self, x, apply_grl=True):
        if self.training and apply_grl:
            self.iter_num += 1

        coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter) if apply_grl else 0
        if apply_grl:
            x.register_hook(grl_hook(coeff))

        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)  # logits，不加 softmax
        return x  # 返回 raw logits（配合 CrossEntropyLoss）

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": self.lr_mult}]


class MultiViewDFC(nn.Module):
    def __init__(self, input_dims, hidden_dim, num_clusters, hidden_size,num_groups, max_iter, lr_mult):
        super(MultiViewDFC, self).__init__()
        self.autoencoder = MultiViewAutoEncoder(input_dims, hidden_dim)
        self.cluster_assignment = ClusterAssignment(num_clusters, input_dims, hidden_dim)
        self.adv_net = AdversarialNetwork(hidden_dim * len(input_dims), hidden_size, num_groups, max_iter, lr_mult)
        self.num_clusters = num_clusters

    def init_kmeans_for_views(self, train_loader, device):
        print("🚀 初始化 K-Means 聚类中心 (每个视图单独计算)")
        with torch.no_grad():
            all_Z_views = None
            for batch in train_loader:
                X_batch, _, _ = batch
                X_batch = [x.to(device) for x in X_batch]
                Z_views = self.autoencoder.encoder(X_batch)
                if all_Z_views is None:
                    all_Z_views = [z.cpu() for z in Z_views]
                else:
                    all_Z_views = [torch.cat((all_Z_views[i], Z_views[i].cpu()), dim=0) for i in range(len(Z_views))]

            self.cluster_assignment.init_cluster_centers_for_views(all_Z_views)
            print("✅ K-Means 初始化完成！")

    def forward(self, X, use_grl=False):
        Z, X_recon = self.autoencoder(X)
        Z_fused = torch.cat(Z, dim=1)

        # ✅ 调用 `ClusterAssignment`，同时返回 `P_fused` 和 `P_views`
        P_fused, P_views = self.cluster_assignment(Z_fused, Z)

        # ✅ 计算 `D` 的输出（对抗网络）
        adv_out = self.adv_net(Z_fused, apply_grl=use_grl)

        # ✅ 现在返回 `Z_fused`
        return P_fused, P_views, Z_fused, adv_out, X_recon


