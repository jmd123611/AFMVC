import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io
import numpy as np
import os

import torch
from torch.utils.data import Dataset
import scipy.io
import numpy as np
import os

class MultiViewDataset(Dataset):
    """
    多视图数据集加载（支持 .mat 格式）
    """
    def __init__(self, data_path, dataset_name, sensitive_attribute="G", return_labels=True):
        """
        :param data_path: 数据文件所在路径（包含 .mat 文件）
        :param dataset_name: 选择的数据集（用于匹配 .mat 文件名）
        :param sensitive_attribute: 受保护属性的 key（默认为 "G"）
        :param return_labels: 是否返回类别标签 Y（默认为 True）
        """
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.sensitive_attribute = sensitive_attribute
        self.return_labels = return_labels  # 是否返回 Y

        # 读取数据
        mat_file = os.path.join(data_path, f"{dataset_name}.mat")
        mat_data = scipy.io.loadmat(mat_file)

        # 解析多视图数据
        self.views = []
        for key in sorted(mat_data.keys()):
            if key.startswith("view_"):  # 确保是视图数据
                self.views.append(mat_data[key].astype(np.float32))

        self.num_views = len(self.views)  # 视图数

        # 读取类别标签 Y
        if "y" in mat_data:
            self.labels = mat_data["y"].astype(np.int64).flatten()
        elif "Y" in mat_data:
            self.labels = mat_data["Y"].astype(np.int64).flatten()
        else:
            raise ValueError("数据集中缺少类别标签 'Y'")

        # 读取受保护属性 G
        if self.sensitive_attribute in mat_data:
            self.sensitive_attr = mat_data[self.sensitive_attribute].astype(np.float32).flatten()
        else:
            raise ValueError(f"数据集中缺少受保护属性 '{self.sensitive_attribute}'")

        # 归一化每个视图
        self.views = [(v - np.mean(v, axis=0)) / (np.std(v, axis=0) + 1e-9) for v in self.views]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        返回：
        - `X`: 多视图数据 `[V1[idx], ..., Vn[idx]]`
        - `G`: 受保护属性
        - `Y`: 类别标签（如果 return_labels=True）
        """
        sample = [torch.tensor(view[idx]) for view in self.views]  # 每个视图的数据
        sensitive_attr = torch.tensor(self.sensitive_attr[idx], dtype=torch.float)  # 受保护属性
        label = torch.tensor(self.labels[idx], dtype=torch.long)  # 类别标签

        if self.return_labels:
            return sample, sensitive_attr, label
        return sample, sensitive_attr



def load_data(dataset_name, data_root="D:/learning/MVC_exp/data/fair/"):
    """
    加载多视图数据集
    :param dataset_name: 选择的数据集（如 "MSRCV1"）
    :param data_root: 数据存放的根目录
    :return: (dataset, dims, num_views, data_size, class_num, data_loader)
    """
    dataset_mapping = {
        "mfeat": {"dims": [216, 76, 64, 6, 240, 47], "num_views": 6, "size": 2000, "classes": 10},
        "Credit_5000": {"dims": [22,22], "num_views": 2, "size": 5000, "classes": 5},
        "COIL": {"dims": [1024, 3304,6750], "num_views": 3, "size": 1440, "classes": 20},
        "bank": {"dims": [12, 12], "num_views": 2, "size": 2907, "classes": 2},
        "law_school_10000": {"dims": [10, 10], "num_views": 2, "size": 10000, "classes": 2}
    }

    if dataset_name not in dataset_mapping:
        raise ValueError(f"未知数据集 '{dataset_name}'，请检查 `dataset_mapping`")

    dataset = MultiViewDataset(data_root, dataset_name)
    dims = dataset_mapping[dataset_name]["dims"]
    num_views = dataset_mapping[dataset_name]["num_views"]
    data_size = dataset_mapping[dataset_name]["size"]
    class_num = dataset_mapping[dataset_name]["classes"]

    data_loader = DataLoader(dataset, batch_size=data_size, shuffle=False)

    return dataset, dims, num_views, data_size, class_num, data_loader
