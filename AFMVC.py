import argparse           #q消融实验 去掉公平性损失！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
import torch
import torch.nn.functional as F
import pandas as pd
from module1multi import MultiViewDFC  # 导入模型
from utils import set_seed, target_distribution  # 工具函数
from eval import purity_score, fairness_evaluation, balance_score, ACC  # 评估方法
from dataloader import load_data  # 加载数据
from sklearn.metrics.cluster import normalized_mutual_info_score
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.cluster import KMeans
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# ====== 解析命令行参数 ======
parser = argparse.ArgumentParser(description="Multi-View Fair Clustering Training")
parser.add_argument("--dataset", type=str, default="mfeat", help="选择数据集")
parser.add_argument("--hidden_dim", type=int, default=64, help="隐藏层维度")
parser.add_argument("--batch_size", type=int, default=256, help="批量大小")
parser.add_argument("--epochs_ae", type=int, default=100, help="自编码器训练轮数")
parser.add_argument("--epochs_dfc", type=int, default=1000, help="公平聚类训练轮数")
parser.add_argument("--T_update", type=int, default=50, help="伪标签更新间隔")
parser.add_argument("--learning_rate", type=float, default=5e-4, help="学习率")
parser.add_argument("--alpha_fair", type=float, default=0.01, help="公平性损失系数")
parser.add_argument("--alpha_recon", type=float, default=1, help="重建损失系数")
parser.add_argument("--alpha_entropy", type=float, default=0.1, help="交叉熵聚类损失系数")
parser.add_argument("--hidden_size", type=int, default=32, help="对抗网络隐藏层大小")
parser.add_argument("--max_iter", type=int, default=20000, help="对抗网络最大迭代次数")
parser.add_argument("--lr_mult", type=float, default=0.5, help="对抗网络学习率倍率")

args = parser.parse_args()


print(f"📌 使用数据集: {args.dataset}")

# ====== 训练超参数 ======
BATCH_SIZE = args.batch_size
NUM_EPOCHS_AE = args.epochs_ae
NUM_EPOCHS_DFC = args.epochs_dfc
LEARNING_RATE = args.learning_rate
ALPHA_FAIR = args.alpha_fair
ALPHA_ENTROPY = args.alpha_entropy
ALPHA_RECON = args.alpha_recon
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HIDDEN_SIZE = args.hidden_size
MAX_ITER = args.max_iter
LR_MULT = args.lr_mult
T_update = args.T_update

# ====== 加载数据 ======
dataset, input_dims, num_views, data_size, num_clusters, train_loader = load_data(args.dataset)
num_sen=2
# 设置 `hidden_dim` 默认值
hidden_dim = args.hidden_dim if args.hidden_dim != -1 else int(sum(input_dims) / len(input_dims))

# 初始化模型
model = MultiViewDFC(input_dims, hidden_dim, num_clusters, hidden_size=HIDDEN_SIZE,num_groups=num_sen, max_iter=MAX_ITER, lr_mult=LR_MULT).to(DEVICE)


# ====== 数据检查函数 ======
def test_dataloader(dataset_name):
    """
    检查 DataLoader 是否正确加载数据
    """
    dataset, dims, num_views, data_size, class_num, train_loader = load_data(dataset_name)

    for X_batch, G_batch, Y_batch in train_loader:
        print(f"✅ 成功加载数据集 '{dataset_name}'，共有 {num_views} 个视图")
        for i in range(num_views):
            print(f"视图 {i+1} 形状: {X_batch[i].shape}")
        print(f"受保护属性 G 形状: {G_batch.shape}")
        print(f"类别标签 Y 形状: {Y_batch.shape}")
        break  # 只打印一个 batch


# 运行数据检查
test_dataloader(args.dataset)


# ====== 训练自编码器 ======
def train_autoencoder(model, train_loader, num_epochs=NUM_EPOCHS_AE):
    """
    训练 `AutoEncoder` 以学习数据的表示。
    """
    optimizer = optim.Adam(model.autoencoder.parameters(), lr=LEARNING_RATE)
    mse_loss_function = nn.MSELoss()

    print("=== 开始训练 AutoEncoder ===")
    for epoch in range(num_epochs):
        total_recon_loss = 0
        for batch in train_loader:
            X_batch, _, _ = batch
            X_batch = [x.to(DEVICE) for x in X_batch]

            optimizer.zero_grad()
            _, X_recon = model.autoencoder(X_batch)

            # 计算重构损失
            recon_loss = sum(mse_loss_function(x, x_recon) for x, x_recon in zip(X_batch, X_recon))

            # 反向传播
            recon_loss.backward()
            optimizer.step()

            total_recon_loss += recon_loss.item()

        avg_recon_loss = total_recon_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] ")

    print("=== 预训练 AutoEncoder 结束，开始 K-Means 计算伪标签 ===")

def train_fair_clustering(model, train_loader, alpha_fair, alpha_entropy, num_epochs=NUM_EPOCHS_DFC,
                          ):
    print("当前参数组合：", alpha_fair, alpha_entropy)

    def compute_kmeans_labels():
        """使用当前 `Z_fused` 计算 `con_labels` 并更新 `ClusterAssignment`"""
        encoded_views_per_encoder = [[] for _ in model.autoencoder.encoder.encoders]
        with torch.no_grad():
            for batch in train_loader:
                X_batch, _, _ = batch
                X_batch = [x.to(DEVICE) for x in X_batch]
                encoded_views = model.autoencoder.encoder(X_batch)
                for i in range(len(encoded_views)):
                    encoded_views_per_encoder[i].append(encoded_views[i].cpu().numpy())
        all_encoded_views = [np.vstack(encoded_views) for encoded_views in encoded_views_per_encoder]
        concatenated_Z = np.hstack(all_encoded_views)

        kmeans = KMeans(n_clusters=num_clusters, n_init=10)
        con_labels = torch.tensor(kmeans.fit_predict(concatenated_Z), dtype=torch.long, device=DEVICE)

        model.cluster_assignment.cluster_centers_fused.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32,
                                                                           device=DEVICE)
        return con_labels

    con_labels = compute_kmeans_labels()
    optimizer_E = optim.Adam([
        {"params": model.autoencoder.encoder.parameters(), "lr": LEARNING_RATE},
        {"params": model.autoencoder.decoder.parameters(), "lr": LEARNING_RATE},
        {"params": model.cluster_assignment.parameters(), "lr": LEARNING_RATE},
        {"params": model.adv_net.parameters(), "lr": LEARNING_RATE * model.adv_net.lr_mult}
    ])

    for epoch in range(num_epochs):
        if  epoch % T_update == 0:
            con_labels = compute_kmeans_labels()

        total_recon_loss = 0
        total_entropy_loss = 0
        total_fair_loss = 0
        total_samples = 0

        for batch in train_loader:
            X_batch, G_batch, _ = batch
            X_batch = [x.to(DEVICE) for x in X_batch]
            G_batch = G_batch.to(DEVICE).float().unsqueeze(1)

            optimizer_E.zero_grad()
            P_fused, P_views, Z_fused, adv_out, X_recon = model(X_batch, use_grl=True)

            ce_loss_function = nn.CrossEntropyLoss()
            recon_loss = sum(F.mse_loss(x, x_recon) for x, x_recon in zip(X_batch, X_recon))

            fair_loss = ce_loss_function(adv_out, G_batch.squeeze(1).long())

            con_labels_onehot = F.one_hot(con_labels, num_classes=num_clusters).float()
            entropy_loss = sum(
                F.kl_div(torch.log(P_view + 1e-8), con_labels_onehot, reduction='batchmean')
                for P_view in P_views
            )
            total_loss = ALPHA_RECON * recon_loss + alpha_fair * fair_loss + alpha_entropy * entropy_loss

            total_loss.backward()
            optimizer_E.step()
            total_recon_loss += recon_loss.item()
            total_fair_loss += fair_loss.item()
            total_entropy_loss += entropy_loss.item()
            total_samples += G_batch.size(0)
        print(
            f"Epoch {epoch + 1}/{num_epochs}  ")

    print("✅ 训练完成！")


# ====== 评估模型 ======
def eval_model(model, train_loader):
    """
    评估模型的 Purity、Fairness 和 Balance 指标
    """
    print("=== 正在评估模型 ===")
    all_preds, all_labels, all_groups = [], [], []

    with torch.no_grad():
        for batch in train_loader:
            X_batch, G_batch, Y_batch = batch
            X_batch = [x.to(DEVICE) for x in X_batch]

            # ✅ 正确解包
            _, P_views, Z_fused, _, _ = model(X_batch)

            # ✅ 把 Z_fused 移动到 CPU
            Z_fused_cpu = Z_fused.detach().cpu().numpy()

            # ✅ 在 Z_fused 上应用 K-means
            kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=0)
            preds = kmeans.fit_predict(Z_fused_cpu)

            # ✅ 收集预测结果
            all_preds.append(preds)
            all_labels.append(Y_batch.cpu().numpy())
            all_groups.append(G_batch.cpu().numpy())

    balance= balance_score(np.concatenate(all_preds), np.concatenate(all_groups))
    acc = ACC(np.concatenate(all_labels), np.concatenate(all_preds))
    nmi = normalized_mutual_info_score(np.concatenate(all_labels), np.concatenate(all_preds))

    print(f" Balance: {balance:.4f} | ACC: {acc:.4f} | NMI: {nmi:.4f}")

    return balance,acc,nmi

# ====== 运行训练 ======
if __name__ == "__main__":
    for run in range(1):
        model = MultiViewDFC(input_dims, hidden_dim, num_clusters, hidden_size=HIDDEN_SIZE,num_groups=num_sen, max_iter=MAX_ITER,
                             lr_mult=LR_MULT).to(DEVICE)
        train_autoencoder(model, train_loader)
        train_fair_clustering(model, train_loader,alpha_fair=ALPHA_FAIR,alpha_entropy=ALPHA_ENTROPY)
        balance, acc, nmi = eval_model(model, train_loader)



