"""
赛马娘AI训练框架 - 训练循环

参考 UmaAi 的 training/train.py，实现训练循环。
- 策略用交叉熵损失
- 价值用Huber Loss
- 支持断点续训
- 自动保存最佳模型
"""

import os
import time
import copy
import random
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from .dataset import UmaTrainDataset, SelfPlayDataset, generate_random_data
from model.network import create_model, load_model, save_model, MODEL_DICT
from config import (
    NN_OUTPUT_C, NN_OUTPUT_C_POLICY, NN_OUTPUT_C_VALUE,
    SEED, LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE, MAX_EPOCHS,
    VALUE_LOSS_WEIGHT_MEAN, VALUE_LOSS_WEIGHT_STDEV, VALUE_LOSS_WEIGHT_OPTIMISTIC,
    VALUE_MEAN_OFFSET, VALUE_MEAN_SCALE, VALUE_STDEV_SCALE,
    SAVE_STEP, INFO_STEP,
    GRAD_CLIP_NORM, LR_SCHEDULER_TYPE, LR_COSINE_ETA_MIN_RATIO,
    LR_STEP_STEP, LR_STEP_GAMMA, EARLY_STOP_PATIENCE,
    VALUE_LOSS_BASE_WEIGHT,
    V_LOSS_ROLLBACK_THRESHOLD, P_LOSS_ROLLBACK_THRESHOLD,
)



def seed_everything(seed: int):
    """设置全局随机种子，确保可复现"""
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def cross_entropy_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """交叉熵损失（用于策略）
    
    参考 UmaAi 的 cross_entropy_loss。
    """
    t = torch.log_softmax(output, dim=1)
    losses = torch.sum(-t * target, dim=1) + torch.sum(torch.log(target + 1e-10) * target, dim=1)
    return losses.mean(dim=0)


def calculate_loss(output: torch.Tensor, label: torch.Tensor):
    """计算损失
    
    参考 UmaAi 的 calculateLoss：
    - 价值损失：3个Huber Loss（平均分、标准差、乐观分）
    - 策略损失：交叉熵
    
    Args:
        output: 模型输出 (batch, NN_OUTPUT_C)
        label: 标签 (batch, NN_OUTPUT_C)
        
    Returns:
        (value_loss, policy_loss)
    """
    output_policy = output[:, :NN_OUTPUT_C_POLICY]
    output_value = output[:, NN_OUTPUT_C_POLICY:]
    label_policy = label[:, :NN_OUTPUT_C_POLICY]
    label_value = label[:, NN_OUTPUT_C_POLICY:]
    
    huber_loss = nn.HuberLoss(reduction='mean', delta=1.0)
    
    # 价值损失（归一化后的）
    v_loss1 = VALUE_LOSS_WEIGHT_MEAN * huber_loss(
        output_value[:, 0],
        (label_value[:, 0] - VALUE_MEAN_OFFSET) / VALUE_MEAN_SCALE
    )
    v_loss2 = VALUE_LOSS_WEIGHT_STDEV * huber_loss(
        output_value[:, 1],
        label_value[:, 1] / VALUE_STDEV_SCALE
    )
    v_loss3 = VALUE_LOSS_WEIGHT_OPTIMISTIC * huber_loss(
        output_value[:, 2],
        (label_value[:, 2] - VALUE_MEAN_OFFSET) / VALUE_MEAN_SCALE
    )
    v_loss = v_loss1 + v_loss2 + v_loss3
    
    # 策略损失
    p_loss = cross_entropy_loss(output_policy, label_policy)
    
    return v_loss, p_loss


def train(
    train_data_path: str,
    val_data_path: str = "",
    model_type: str = "ems",
    model_param: tuple = None,
    save_name: str = "auto",
    batch_size: int = BATCH_SIZE,
    lr_scale: float = 1.0,
    wd_scale: float = 1.0,
    max_step: int = 500000,
    max_epochs: int = MAX_EPOCHS,
    save_step: int = SAVE_STEP,
    info_step: int = INFO_STEP,
    gpu: int = 0,
    new_train: bool = False,
    rollback_threshold: float = 0.05,
    grad_clip_norm: float = GRAD_CLIP_NORM,
    lr_scheduler_type: str = LR_SCHEDULER_TYPE,
    early_stop_patience: int = EARLY_STOP_PATIENCE,
):
    """训练主循环
    
    Args:
        train_data_path: 训练数据路径（npz文件或目录）
        val_data_path: 验证数据路径
        model_type: 模型类型
        model_param: 模型参数
        save_name: 保存名称
        batch_size: 批大小
        lr_scale: 学习率缩放
        wd_scale: 权重衰减缩放
        max_step: 最大训练步数
        max_epochs: 最大epoch数
        save_step: 保存间隔
        info_step: 日志间隔
        gpu: GPU编号（-1为CPU）
        rollback_threshold: 回滚阈值
        new_train: 是否重新训练
        grad_clip_norm: 梯度裁剪L2范数阈值，0表示不裁剪
        lr_scheduler_type: 学习率调度器类型，"cosine"/"step"/空字符串
        early_stop_patience: 早停耐心值，0表示不早停
    """
    # 设置随机种子，确保可复现
    seed_everything(SEED)

    # 设备
    device = torch.device(f"cuda:{gpu}" if gpu >= 0 and torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 保存名称
    if save_name == "auto":
        save_name = model_type
        if model_param:
            for p in model_param:
                save_name += f"_{p}"

    # 保存目录
    base_path = f"./saved_models/{save_name}/"
    os.makedirs(base_path, exist_ok=True)
    backup_path = os.path.join(base_path, "backup")
    os.makedirs(backup_path, exist_ok=True)
    model_path = os.path.join(base_path, "model.pth")

    # 加载或创建模型
    loaded_optimizer_state = None
    if os.path.exists(model_path) and not new_train and save_name != "null":
        model = load_model(model_path, device=str(device))
        model_data = torch.load(model_path, map_location="cpu")
        total_step = model_data['totalstep']
        model_type_loaded = model_data['model_type']
        model_param_loaded = model_data['model_param']
        # 恢复优化器状态（如果有）
        if 'optimizer_state_dict' in model_data:
            loaded_optimizer_state = model_data['optimizer_state_dict']
        print(f"加载模型: type={model_type_loaded}, param={model_param_loaded}, step={total_step}")
    else:
        total_step = 0
        model = create_model(model_type, model_param).to(device)

    start_step = total_step
    model.train()

    # 优化器
    lr = LEARNING_RATE * lr_scale
    wd = WEIGHT_DECAY * wd_scale
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    # 断点续训时恢复优化器状态
    if loaded_optimizer_state is not None:
        optimizer.load_state_dict(loaded_optimizer_state)
        print("已恢复优化器状态")

    # 学习率调度器
    scheduler = None
    if lr_scheduler_type == "cosine":
        eta_min = lr * LR_COSINE_ETA_MIN_RATIO
        scheduler = CosineAnnealingLR(optimizer, T_max=max_step, eta_min=eta_min)
    elif lr_scheduler_type == "step":
        scheduler = StepLR(optimizer, step_size=LR_STEP_STEP, gamma=LR_STEP_GAMMA)

    # 早停状态
    best_val_loss = None
    early_stop_counter = 0

    # 回滚备份（同时备份模型和优化器状态）
    model_backup1 = copy.deepcopy(model.state_dict())
    model_backup2 = copy.deepcopy(model.state_dict())
    opt_backup1 = copy.deepcopy(optimizer.state_dict())
    opt_backup2 = copy.deepcopy(optimizer.state_dict())
    backup1_step = start_step
    backup2_step = start_step
    backup1_loss = 1e10
    backup2_loss = 1e10
    v_backup1_loss = 1e10  # v_loss备份值（用于分别判断v_loss是否爆炸）
    v_backup2_loss = 1e10
    p_backup1_loss = 1e10  # p_loss备份值（用于分别判断p_loss是否爆炸）
    p_backup2_loss = 1e10

    # 训练数据
    print("加载训练数据...")
    t_dataset = UmaTrainDataset(train_data_path)
    t_dataloader = DataLoader(t_dataset, shuffle=True, batch_size=batch_size)
    print(f"训练数据: {len(t_dataset)} 条")

    # 验证数据
    v_dataset = None
    if val_data_path and os.path.exists(val_data_path):
        v_dataset = UmaTrainDataset(val_data_path)
        print(f"验证数据: {len(v_dataset)} 条")

    # 训练循环：显式epoch管理
    print("开始训练...")
    time0 = time.time()
    loss_record = [0, 0, 0, 1e-30, 0, 0]  # total, v, p, count, acc, diff

    for epoch in range(max_epochs):
        # epoch级累计
        epoch_loss_sum = 0.0
        epoch_step_count = 0

        for _, (x, label) in enumerate(t_dataloader):
            if x.shape[0] != batch_size:
                continue

            x = x.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            nn_output = model(x)

            v_loss, p_loss = calculate_loss(nn_output, label)

            # 策略精度计算
            output_policy = nn_output[:, :NN_OUTPUT_C_POLICY]
            label_policy = label[:, :NN_OUTPUT_C_POLICY]
            _, p1_predicted = torch.max(output_policy, 1)
            p1_label_values, p1_labels = torch.max(label_policy, 1)
            p1_predicted_values = label_policy[torch.arange(label_policy.shape[0]), p1_predicted]
            p1_correct = (p1_predicted == p1_labels).sum().item()

            # value_loss始终参与，用固定权重（不再随机丢弃）
            loss = p_loss + VALUE_LOSS_BASE_WEIGHT * v_loss

            loss_record[0] += (v_loss.detach().item() + p_loss.detach().item())
            loss_record[1] += v_loss.detach().item()
            loss_record[2] += p_loss.detach().item()
            loss_record[3] += 1
            loss_record[4] += p1_correct / batch_size

            # epoch级累计
            epoch_loss_sum += loss.item()
            epoch_step_count += 1

            loss.backward()
            # 梯度裁剪
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            # 学习率调度
            if scheduler is not None:
                scheduler.step()

            total_step += 1

            # 日志
            if total_step % info_step == 0:
                time1 = time.time()
                time_used = time1 - time0
                time0 = time1

                total_loss_train = loss_record[0] / loss_record[3]
                v_loss_train = loss_record[1] / loss_record[3]
                p_loss_train = loss_record[2] / loss_record[3]
                p1_acc_train = loss_record[4] / loss_record[3]

                print(f"step={total_step}, epoch={epoch}, time={time_used:.1f}s, "
                      f"total_loss={total_loss_train:.4f}, "
                      f"v_loss={v_loss_train:.4f}(bk={v_backup1_loss:.4f}), "
                      f"p_loss={p_loss_train:.4f}(bk={p_backup1_loss:.4f}), "
                      f"p1_acc={100*p1_acc_train:.1f}%, "
                      f"lr={optimizer.param_groups[0]['lr']:.6f}")

                # 分别检查v_loss和p_loss是否爆炸
                v_rollback = V_LOSS_ROLLBACK_THRESHOLD > 0 and v_loss_train > v_backup1_loss + V_LOSS_ROLLBACK_THRESHOLD
                p_rollback = P_LOSS_ROLLBACK_THRESHOLD > 0 and p_loss_train > p_backup1_loss + P_LOSS_ROLLBACK_THRESHOLD
                total_rollback = total_loss_train > backup1_loss + rollback_threshold

                if v_rollback or p_rollback or total_rollback:
                    # 打印是哪个loss爆了
                    reasons = []
                    if v_rollback:
                        reasons.append(f"v_loss({v_loss_train:.4f}>{v_backup1_loss:.4f}+{V_LOSS_ROLLBACK_THRESHOLD})")
                    if p_rollback:
                        reasons.append(f"p_loss({p_loss_train:.4f}>{p_backup1_loss:.4f}+{P_LOSS_ROLLBACK_THRESHOLD})")
                    if total_rollback:
                        reasons.append(f"total({total_loss_train:.4f}>{backup1_loss:.4f}+{rollback_threshold})")
                    print(f"Loss爆炸({', '.join(reasons)})，回滚到step {backup2_step}")
                    model.load_state_dict(model_backup2)
                    optimizer.load_state_dict(opt_backup2)
                    model_backup1 = copy.deepcopy(model_backup2)
                    opt_backup1 = copy.deepcopy(opt_backup2)
                    total_step = backup2_step
                    backup1_step = backup2_step
                    backup1_loss = backup2_loss
                    v_backup1_loss = v_backup2_loss
                    p_backup1_loss = p_backup2_loss
                else:
                    model_backup2 = model_backup1
                    opt_backup2 = opt_backup1
                    backup2_step = backup1_step
                    backup2_loss = backup1_loss
                    v_backup2_loss = v_backup1_loss
                    p_backup2_loss = p_backup1_loss
                    model_backup1 = copy.deepcopy(model.state_dict())
                    opt_backup1 = copy.deepcopy(optimizer.state_dict())
                    backup1_step = total_step
                    backup1_loss = total_loss_train
                    v_backup1_loss = v_loss_train
                    p_backup1_loss = p_loss_train

                loss_record = [0, 0, 0, 1e-30, 0, 0]

            # 保存和验证
            if total_step % save_step == 0 or total_step - start_step >= max_step:
                print(f"保存模型 step={total_step}")
                save_model(model, model_path, total_step, optimizer=optimizer)

                # 验证
                if v_dataset is not None:
                    print("验证中...")
                    v_dataloader = DataLoader(v_dataset, shuffle=False, batch_size=batch_size)
                    v_loss_record = [0, 0, 0, 1e-30, 0]
                    model.eval()

                    with torch.no_grad():
                        for x, label in v_dataloader:
                            if x.shape[0] != batch_size:
                                continue
                            x = x.to(device)
                            label = label.to(device)

                            nn_output = model(x)
                            v_loss, p_loss = calculate_loss(nn_output, label)
                            v_loss_record[0] += (v_loss.item() + p_loss.item())
                            v_loss_record[1] += v_loss.item()
                            v_loss_record[2] += p_loss.item()
                            v_loss_record[3] += 1

                    if v_loss_record[3] > 0:
                        total_val = v_loss_record[0] / v_loss_record[3]
                        v_val = v_loss_record[1] / v_loss_record[3]
                        p_val = v_loss_record[2] / v_loss_record[3]
                        print(f"验证: total={total_val:.4f}, v={v_val:.4f}, p={p_val:.4f}")

                        # 早停检查
                        if early_stop_patience > 0:
                            if best_val_loss is None or total_val < best_val_loss:
                                best_val_loss = total_val
                                early_stop_counter = 0
                            else:
                                early_stop_counter += 1
                                print(f"早停计数: {early_stop_counter}/{early_stop_patience} (最佳验证loss={best_val_loss:.4f})")
                                if early_stop_counter >= early_stop_patience:
                                    print(f"早停触发！连续{early_stop_patience}个save_step验证loss未下降，停止训练")
                                    save_model(model, model_path, total_step, optimizer=optimizer)
                                    return

                    model.train()

            # total_step退出条件
            if total_step - start_step >= max_step:
                print("训练完成! (达到max_step)")
                return

        # epoch结束日志
        if epoch_step_count > 0:
            epoch_avg_loss = epoch_loss_sum / epoch_step_count
            print(f"[Epoch {epoch}] 完成, steps={epoch_step_count}, "
                  f"avg_loss={epoch_avg_loss:.4f}, "
                  f"total_step={total_step}, "
                  f"lr={optimizer.param_groups[0]['lr']:.6f}")

        # epoch结束后也检查total_step退出
        if total_step - start_step >= max_step:
            print("训练完成! (达到max_step)")
            return

    print(f"训练完成! (达到max_epochs={max_epochs})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="赛马娘AI训练")
    parser.add_argument('--train_data', type=str, default='../data/train.npz')
    parser.add_argument('--val_data', type=str, default='')
    parser.add_argument('--model_type', type=str, default='ems')
    parser.add_argument('--model_param', nargs='+', type=int, default=None)
    parser.add_argument('--save_name', type=str, default='auto')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--new', action='store_true', default=False)
    parser.add_argument('--max_step', type=int, default=500000)
    parser.add_argument('--max_epochs', type=int, default=MAX_EPOCHS)
    args = parser.parse_args()

    model_param = tuple(args.model_param) if args.model_param else None

    train(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        model_type=args.model_type,
        model_param=model_param,
        save_name=args.save_name,
        batch_size=args.batch_size,
        gpu=args.gpu,
        new_train=args.new,
        max_step=args.max_step,
        max_epochs=args.max_epochs,
    )
