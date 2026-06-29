"""
赛马娘AI训练框架 - MCTS搜索参数配置

包含搜索阶段配置、搜索常量、默认搜索参数、
策略软化温度、手写评估参数、NN校准开关。
"""

# ============================================================================
# MCTS搜索参数
# ============================================================================

# 搜索阶段配置（先粗后精，淘汰低分选项）
SEARCH_STAGE_NUM = 3
SEARCH_FACTOR_STAGE = [0.25, 0.25, 0.5]
SEARCH_THRESHOLD_STDEV_STAGE = [4, 4, 0]

# 搜索常量
EXPECTED_SEARCH_STDEV = 2200
NORM_DISTRIBUTION_SAMPLING = 128  # 正态分布采样点数

# 默认搜索参数
SEARCH_SINGLE_MAX = 256  # 单个动作最大搜索次数
SEARCH_TOTAL_MAX = 0  # 总搜索次数限制（0为不限）
SEARCH_GROUP_SIZE = 128  # 每组搜索量
SEARCH_CPUT = 1.0  # cpuct参数
SEARCH_MAX_DEPTH = 10  # 蒙特卡洛深度
SEARCH_MAX_RADICAL_FACTOR = 5.0  # 最大激进度

# 策略软化温度参数（修复P1-11：正确的温度softmax，替代错误的exp(x/N/delta)）
MCTS_POLICY_TEMPERATURE = 1.0  # 策略温度，越大越均匀；1.0为标准softmax

# ============================================================================
# MCTS决策归因日志参数
# ============================================================================

# 手写评估默认标准差（用于非NN评估时的stdev估算）
# 原来硬编码在_evaluate_game和selfplay里，现统一到config
HANDWRITTEN_STDEV_BASE = 500.0    # 手写评估初始标准差
HANDWRITTEN_STDEV_FLOOR = 100.0   # 手写评估最终标准差（游戏接近结束时）

# NN vs 手写对账模式（默认关闭，仅诊断用）
COMPARE_WITH_HANDWRITTEN = False

# 搜索结束后是否用batch NN校准root动作value（P0-2）
# 开启后，MCTS搜索结束会对root每个合法动作做一次batch NN推理，
# 将NN评估结果作为额外样本校准搜索结果
MCTS_BATCH_NN_CALIBRATE = True
