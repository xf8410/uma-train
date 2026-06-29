"""
赛马娘AI训练框架 - 训练超参与自我对弈参数配置

包含学习率、批大小、损失权重、保存间隔、学习率调度、
早停、自我对弈参数、训练损失与rollback参数。
"""

# ============================================================================
# 训练超参数
# ============================================================================

# 随机种子
SEED = 42

# 学习率
LEARNING_RATE = 7e-4
WEIGHT_DECAY = 1e-5

# 批大小
BATCH_SIZE = 1024

# 价值损失权重
VALUE_LOSS_WEIGHT_MEAN = 0.2
VALUE_LOSS_WEIGHT_STDEV = 0.4
VALUE_LOSS_WEIGHT_OPTIMISTIC = 0.2

# 价值归一化参数
VALUE_MEAN_OFFSET = 38000
VALUE_MEAN_SCALE = 300
VALUE_STDEV_SCALE = 150

# 保存间隔
SAVE_STEP = 2000
INFO_STEP = 500

# 梯度裁剪
GRAD_CLIP_NORM = 1.0  # L2范数裁剪阈值，0表示不裁剪

# 学习率调度
LR_SCHEDULER_TYPE = "cosine"  # 调度器类型："cosine"或"step"，空字符串表示不用
LR_COSINE_ETA_MIN_RATIO = 0.05  # 余弦退火最小学习率比例
LR_STEP_STEP = 50000  # 阶梯衰减步数
LR_STEP_GAMMA = 0.1  # 阶梯衰减倍率

# 早停
EARLY_STOP_PATIENCE = 0  # 早停耐心值，连续N个save_step验证loss不降就停，0表示不早停

# ============================================================================
# 自我对弈参数
# ============================================================================

SELFPLAY_NUM_GAMES = 100  # 自我对弈局数
SELFPLAY_BATCH_SIZE = 16  # 自我对弈批大小

# ============================================================================
# 训练损失与rollback参数
# ============================================================================

# value_loss基础权重（始终参与loss计算，不再随机丢弃）
# 原来用value_sampling随机丢弃value_loss，改为固定权重
VALUE_LOSS_BASE_WEIGHT = 0.5

# v_loss和p_loss各自的rollback阈值（相对于最近备份点的增量）
# 设为0表示不单独检查该loss，只看total
V_LOSS_ROLLBACK_THRESHOLD = 0.05
P_LOSS_ROLLBACK_THRESHOLD = 0.05
