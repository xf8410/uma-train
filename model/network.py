"""
赛马娘AI训练框架 - PyTorch模型

参考 UmaAi 的 training/model.py 的 Model_EncoderMlpSimple 架构。
输入：全局状态 + 6张支援卡 + 场上人头信息
输出：策略(53维) + 价值(3维：平均分、标准差、乐观分)

模型大小控制在376KB以内（手机端推理限制）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    NN_INPUT_C, NN_INPUT_C_GLOBAL, NN_INPUT_C_CARD, NN_INPUT_C_CARDPERSON,
    NN_INPUT_C_PERSON,
    NN_CARD_NUM, NN_HEAD_NUM, NN_OUTPUT_C, NN_OUTPUT_C_POLICY, NN_OUTPUT_C_VALUE,
    DEFAULT_ENCODER_BLOCKS, DEFAULT_ENCODER_FEATURES, DEFAULT_MLP_BLOCKS,
    DEFAULT_MLP_FEATURES, DEFAULT_GLOBAL_FEATURES, MODEL_SIZE_LIMIT,
)


class LinearBN(nn.Module):
    """带LayerNorm的线性层（P2-5修复：BatchNorm1d在batch_size=1时崩溃，改用LayerNorm）"""
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.lin = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        self.bn = nn.LayerNorm(out_features)
    
    def forward(self, x):
        y = self.lin(x)
        y = self.bn(y)
        return y


class EncoderLayerSimple(nn.Module):
    """简易版Encoder层（参考UmaAi的EncoderLayerSimple）
    
    使用简化的注意力机制：ReLU(x @ Q^T) @ V
    """
    def __init__(self, inout_c: int, global_c: int):
        super().__init__()
        self.inout_c = inout_c
        self.lin_Q = nn.Linear(inout_c, inout_c, bias=False)
        self.lin_V = nn.Linear(inout_c, inout_c, bias=False)
        self.lin_global = nn.Linear(global_c, inout_c, bias=False)
    
    def forward(self, x, gf):
        """前向传播
        
        Args:
            x: (batch, head_num, inout_c)
            gf: (batch, global_c)
        """
        batch_size, head_num, _ = x.shape
        y = x.reshape(batch_size * head_num, self.inout_c)
        q = self.lin_Q(y).view(batch_size, head_num, self.inout_c)
        v = self.lin_V(y).view(batch_size, head_num, self.inout_c)
        
        # 简化注意力：ReLU(x @ Q^T) / head_num
        att = torch.relu(torch.bmm(x, q.transpose(1, 2))) / head_num
        y = torch.bmm(att, v)
        y = torch.relu(y + self.lin_global(gf).view(batch_size, 1, self.inout_c))
        
        # 残差连接
        y = y + x
        return y


class ResnetLayer(nn.Module):
    """残差MLP层"""
    def __init__(self, inout_c: int, mid_c: int):
        super().__init__()
        self.lin1 = nn.Linear(inout_c, mid_c, bias=False)
        self.lin2 = nn.Linear(mid_c, inout_c, bias=False)
    
    def forward(self, x):
        y = self.lin1(x)
        y = torch.relu(y)
        y = self.lin2(y)
        y = torch.relu(y)
        y = y + x
        return y


class ModelEncoderMlpSimple(nn.Module):
    """EncoderMLP Simple模型（参考UmaAi的Model_EncoderMlpSimple）
    
    架构：
    1. 输入编码：全局信息 -> globalF维，支援卡 -> encoderF维
    2. Encoder：多组简化的自注意力层，全局信息作为条件
    3. MLP：残差MLP层，输出策略和价值
    
    输入格式（NN_HEAD_NUM==0时，人头合并到卡槽）：
    - x[:, :NN_INPUT_C_GLOBAL] - 全局信息（156维）
    - x[:, NN_INPUT_C_GLOBAL:] - 支援卡信息（6卡x38维/卡）
    
    输出格式：
    - output[:, :NN_OUTPUT_C_POLICY] - 策略logits (53维)
    - output[:, NN_OUTPUT_C_POLICY:] - 价值 (3维: 归一化平均分, 归一化标准差, 归一化乐观分)
    """
    
    def __init__(
        self,
        encoder_blocks: int = DEFAULT_ENCODER_BLOCKS,
        encoder_features: int = DEFAULT_ENCODER_FEATURES,
        mlp_blocks: int = DEFAULT_MLP_BLOCKS,
        mlp_features: int = DEFAULT_MLP_FEATURES,
        global_features: int = DEFAULT_GLOBAL_FEATURES,
    ):
        super().__init__()
        self.model_type = "ems"
        self.model_param = (encoder_blocks, encoder_features, mlp_blocks, mlp_features, global_features)
        self.encoder_features = encoder_features
        
        # 输入编码层
        # 修复P0-7：当Head_Num==0时，每个卡槽包含卡片+人头信息，使用CardPerson维度
        self._card_input_dim = NN_INPUT_C_CARDPERSON if NN_HEAD_NUM == 0 else NN_INPUT_C_CARD
        self.inputhead_global1 = nn.Linear(NN_INPUT_C_GLOBAL, global_features, bias=False)
        self.inputhead_global2 = nn.Linear(global_features, encoder_features, bias=False)
        self.inputhead_card = nn.Linear(self._card_input_dim, encoder_features, bias=False)
        
        if NN_HEAD_NUM != 0:
            self.inputhead_person = nn.Linear(NN_INPUT_C_PERSON, encoder_features, bias=False)
        
        # Encoder层
        self.encoder_trunk = nn.ModuleList([
            EncoderLayerSimple(inout_c=encoder_features, global_c=global_features)
            for _ in range(encoder_blocks)
        ])
        
        # 连接层：encoder -> mlp
        self.lin_before_mlp1 = nn.Linear(global_features, mlp_features, bias=False)
        self.lin_before_mlp2 = nn.Linear(encoder_features, mlp_features, bias=False)
        
        # MLP层
        self.mlp_trunk = nn.ModuleList([
            ResnetLayer(mlp_features, mlp_features)
            for _ in range(mlp_blocks)
        ])
        
        # 输出层
        self.outputhead = nn.Linear(mlp_features, NN_OUTPUT_C)
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: (batch, NN_INPUT_C) 输入向量
            
        Returns:
            (batch, NN_OUTPUT_C) 输出向量
        """
        # 分割输入
        x1 = x[:, :NN_INPUT_C_GLOBAL]  # 全局信息
        
        if NN_HEAD_NUM != 0:
            x2 = x[:, NN_INPUT_C_GLOBAL:NN_INPUT_C_GLOBAL + NN_CARD_NUM * NN_INPUT_C_CARD].reshape(-1, NN_INPUT_C_CARD)
            x3 = x[:, NN_INPUT_C_GLOBAL + NN_CARD_NUM * NN_INPUT_C_CARD:].reshape(-1, NN_INPUT_C_PERSON)
        else:
            # 修复P0-7：使用卡槽维度（含人头信息）而非纯卡片维度
            x2 = x[:, NN_INPUT_C_GLOBAL:].reshape(-1, self._card_input_dim)
        
        # 全局特征
        gf = torch.relu(self.inputhead_global1(x1))
        
        # 编码输入
        if NN_HEAD_NUM != 0:
            h = (
                self.inputhead_global2(gf).view(-1, 1, self.encoder_features) +
                F.pad(self.inputhead_card(x2).view(-1, NN_CARD_NUM, self.encoder_features),
                      (0, 0, 0, NN_HEAD_NUM - NN_CARD_NUM, 0, 0)) +
                self.inputhead_person(x3).view(-1, NN_HEAD_NUM, self.encoder_features)
            )
        else:
            h = (
                self.inputhead_global2(gf).view(-1, 1, self.encoder_features) +
                self.inputhead_card(x2).view(-1, NN_CARD_NUM, self.encoder_features)
            )
        
        h = torch.relu(h)
        
        # Encoder层
        for block in self.encoder_trunk:
            h = block(h, gf)
        
        # 池化：对所有head取平均
        h = h.mean(dim=1)
        
        # 连接到MLP
        h = self.lin_before_mlp2(h) + self.lin_before_mlp1(gf)
        h = torch.relu(h)
        
        # MLP层
        for block in self.mlp_trunk:
            h = block(h)
        
        # 输出
        return self.outputhead(h)
    
    def get_model_size_kb(self) -> float:
        """计算模型大小（KB）"""
        total_params = sum(p.numel() for p in self.parameters())
        # float32 = 4 bytes
        size_bytes = total_params * 4
        return size_bytes / 1024


# ============================================================================
# 模型字典
# ============================================================================

class ModelTwoLayer(nn.Module):
    """两层MLP（简单基线）"""
    def __init__(self, mid_c: int = 256):
        super().__init__()
        self.model_type = "tl"
        self.model_param = (mid_c,)
        self.inputhead = nn.Linear(NN_INPUT_C, mid_c)
        self.outputhead = nn.Linear(mid_c, NN_OUTPUT_C)
    
    def forward(self, x):
        y = self.inputhead(x)
        y = torch.relu(y)
        y = self.outputhead(y)
        return y


class ModelLinear(nn.Module):
    """线性模型（最简单基线）"""
    def __init__(self, _=0):
        super().__init__()
        self.model_type = "lin"
        self.model_param = (_,)
        self.linear1 = nn.Linear(NN_INPUT_C, NN_OUTPUT_C)
    
    def forward(self, x):
        return self.linear1(x)


# 模型类型注册表
MODEL_DICT = {
    "ems": ModelEncoderMlpSimple,  # 主要使用的模型
    "tl": ModelTwoLayer,            # 简单基线
    "lin": ModelLinear,             # 线性基线
}


def create_model(model_type: str = "ems", model_param: tuple = None) -> nn.Module:
    """创建模型
    
    Args:
        model_type: 模型类型
        model_param: 模型参数元组
        
    Returns:
        PyTorch模型
    """
    if model_param is None:
        # 默认小模型（满足376KB限制）
        model_param = (1, 96, 2, 192, 192)
    
    model_cls = MODEL_DICT.get(model_type)
    if model_cls is None:
        raise ValueError(f"未知模型类型: {model_type}, 可用: {list(MODEL_DICT.keys())}")
    
    model = model_cls(*model_param)
    
    # 检查模型大小
    size_kb = model.get_model_size_kb() if hasattr(model, 'get_model_size_kb') else 0
    if size_kb > MODEL_SIZE_LIMIT / 1024:
        print(f"警告: 模型大小 {size_kb:.1f}KB 超过限制 {MODEL_SIZE_LIMIT/1024:.1f}KB")
    
    return model


def load_model(path: str, device: str = "cpu") -> nn.Module:
    """加载已保存的模型
    
    Args:
        path: 模型文件路径
        device: 设备
        
    Returns:
        加载的模型
    """
    model_data = torch.load(path, map_location=device)
    model_type = model_data['model_type']
    model_param = model_data['model_param']
    
    model = MODEL_DICT[model_type](*model_param).to(device)
    model.load_state_dict(model_data['state_dict'])
    model.eval()
    
    return model


def save_model(model: nn.Module, path: str, total_step: int = 0, optimizer=None):
    """保存模型
    
    Args:
        model: 要保存的模型
        path: 保存路径
        total_step: 当前训练步数
        optimizer: 可选，优化器（用于断点续训时恢复训练状态）
    """
    checkpoint = {
        'totalstep': total_step,
        'state_dict': model.state_dict(),
        'model_type': model.model_type,
        'model_param': model.model_param,
    }
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    torch.save(checkpoint, path)


if __name__ == "__main__":
    # 简单测试：创建模型并forward一个batch
    print("=== 模型测试 ===")
    
    # 测试默认模型
    model = create_model("ems")
    print(f"模型类型: {model.model_type}")
    print(f"模型参数: {model.model_param}")
    print(f"模型大小: {model.get_model_size_kb():.1f} KB")
    
    # 测试forward
    batch_size = 4
    x = torch.randn(batch_size, NN_INPUT_C)
    
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"策略输出范围: [{output[0, :NN_OUTPUT_C_POLICY].min():.3f}, {output[0, :NN_OUTPUT_C_POLICY].max():.3f}]")
    print(f"价值输出: {output[0, NN_OUTPUT_C_POLICY:].tolist()}")
    print("测试通过!")
