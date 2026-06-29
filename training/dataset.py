"""
赛马娘AI训练框架 - 数据集

参考 UmaAi 的 training/dataset.py，支持npz格式的训练数据。
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import os

from config import NN_INPUT_C, NN_OUTPUT_C, NN_OUTPUT_C_POLICY


class UmaTrainDataset(Dataset):
    """赛马娘训练数据集
    
    支持UmaAi格式的npz文件，包含:
    - x: 神经网络输入 (N, NN_INPUT_C)
    - label: 标签 (N, NN_OUTPUT_C)，前NN_OUTPUT_C_POLICY维为策略，后3维为价值
    """
    
    def __init__(self, npz_path: str, sampling: float = 1.0):
        """
        Args:
            npz_path: npz文件路径
            sampling: 采样率（0-1），用于避免过拟合
        """
        data = np.load(npz_path)
        self.x = data["x"].astype(np.float32)
        self.label = data["label"].astype(np.float32)
        
        # 采样
        if sampling < 1.0:
            n = len(self.x)
            sample_size = int(n * sampling)
            indices = np.random.choice(n, sample_size, replace=False)
            self.x = self.x[indices]
            self.label = self.label[indices]
    
    def __getitem__(self, index):
        return self.x[index], self.label[index]
    
    def __len__(self):
        return self.x.shape[0]


class SelfPlayDataset(Dataset):
    """自我对弈生成的训练数据集
    
    从多个npz文件中加载自我对弈数据。
    """
    
    def __init__(self, data_dir: str, max_files: int = 0):
        """
        Args:
            data_dir: 包含npz文件的目录
            max_files: 最多加载的文件数（0为全部）
        """
        self.x_list = []
        self.label_list = []
        
        files = sorted([
            os.path.join(data_dir, f) 
            for f in os.listdir(data_dir) 
            if f.endswith('.npz')
        ])
        
        if max_files > 0:
            files = files[:max_files]
        
        for f in files:
            data = np.load(f)
            self.x_list.append(data["x"].astype(np.float32))
            self.label_list.append(data["label"].astype(np.float32))
        
        if self.x_list:
            self.x = np.concatenate(self.x_list, axis=0)
            self.label = np.concatenate(self.label_list, axis=0)
        else:
            self.x = np.zeros((0, NN_INPUT_C), dtype=np.float32)
            self.label = np.zeros((0, NN_OUTPUT_C), dtype=np.float32)
    
    def __getitem__(self, index):
        return self.x[index], self.label[index]
    
    def __len__(self):
        return self.x.shape[0]


def generate_random_data(num_samples: int, save_path: str):
    """生成随机训练数据（用于验证训练流程）
    
    Args:
        num_samples: 样本数量
        save_path: 保存路径
    """
    x = np.random.randn(num_samples, NN_INPUT_C).astype(np.float32)
    
    # 随机策略（softmax后的概率）
    policy = np.random.dirichlet(np.ones(NN_OUTPUT_C_POLICY), size=num_samples).astype(np.float32)
    
    # 随机价值
    value_mean = (np.random.randn(num_samples) * 300 + 38000).astype(np.float32)
    value_stdev = (np.abs(np.random.randn(num_samples)) * 150).astype(np.float32)
    value_optimistic = (value_mean + value_stdev).astype(np.float32)
    
    # 归一化价值
    value_mean_norm = ((value_mean - 38000) / 300).astype(np.float32)
    value_stdev_norm = (value_stdev / 150).astype(np.float32)
    value_optimistic_norm = ((value_optimistic - 38000) / 300).astype(np.float32)
    
    label = np.concatenate([
        policy,
        np.stack([value_mean_norm, value_stdev_norm, value_optimistic_norm], axis=1)
    ], axis=1).astype(np.float32)
    
    np.savez(save_path, x=x, label=label)
    print(f"已生成 {num_samples} 条随机训练数据，保存到 {save_path}")
