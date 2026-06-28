"""
赛马娘AI训练框架 - 全局配置

包含输入输出维度、超参数、游戏常量等。
参考 UmaAi 的 training/config.py 和 GameDatabase/GameConstants.h
"""

# ============================================================================
# 游戏基础常量
# ============================================================================

TOTAL_TURN = 78  # 总回合数
MAX_INFO_PERSON_NUM = 6  # 支援卡数量
MAX_PERSON_PER_TRAIN = 5  # 每个训练最多人头数
MAX_SCORE = 60000  # 分数分布数组大小

# 属性上限（初始值），1200以上2的倍数才生效
BASIC_FIVE_STATUS_LIMIT = [2300, 2200, 1800, 1400, 1400]

# 训练类型枚举
TRAIN_SPEED = 0
TRAIN_STAMINA = 1
TRAIN_POWER = 2
TRAIN_GUTS = 3
TRAIN_WIT = 4

TRAIN_NAMES = ["速度", "耐力", "力量", "根性", "智力"]

# 训练基础值 [训练类型][训练等级][速,耐,力,根,智,pt,体力消耗]
# 等级0-4对应训练等级
TRAINING_BASIC_VALUE = [
    # 速度
    [
        [11, 0, 2, 0, 0, 5, -19],
        [12, 0, 2, 0, 0, 5, -20],
        [13, 0, 2, 0, 0, 5, -21],
        [14, 0, 3, 0, 0, 5, -23],
        [15, 0, 4, 0, 0, 5, -25],
    ],
    # 耐力
    [
        [0, 10, 0, 4, 0, 5, -20],
        [0, 11, 0, 4, 0, 5, -21],
        [0, 12, 0, 5, 0, 5, -22],
        [0, 13, 0, 5, 0, 5, -24],
        [0, 14, 0, 6, 0, 5, -26],
    ],
    # 力量
    [
        [0, 4, 10, 0, 0, 5, -20],
        [0, 4, 11, 0, 0, 5, -21],
        [0, 5, 12, 0, 0, 5, -22],
        [0, 5, 13, 0, 0, 5, -24],
        [0, 6, 14, 0, 0, 5, -26],
    ],
    # 根性
    [
        [2, 0, 2, 9, 0, 5, -20],
        [2, 0, 2, 10, 0, 5, -21],
        [2, 0, 2, 11, 0, 5, -22],
        [3, 0, 2, 12, 0, 5, -24],
        [4, 0, 3, 13, 0, 5, -26],
    ],
    # 智力（智力训练体力回复，不消耗）
    [
        [2, 0, 0, 0, 8, 5, 5],
        [2, 0, 0, 0, 9, 5, 5],
        [2, 0, 0, 0, 10, 5, 5],
        [3, 0, 0, 0, 11, 5, 5],
        [4, 0, 0, 0, 12, 5, 5],
    ],
]

# 失败率基础参数 [训练类型][等级]
# 失败率计算: x0 = 0.1 * FailRateBasic[train][level]
# f = (100 - vital) * (x0 - vital) / 40.0 当 vital < x0 时
FAIL_RATE_BASIC = [
    [520, 524, 528, 532, 536],  # 速度
    [507, 511, 515, 519, 523],  # 耐力
    [516, 520, 524, 528, 532],  # 力量
    [532, 536, 540, 544, 548],  # 根性
    [320, 321, 322, 323, 324],  # 智力
]

# 比赛基础奖励
RACE_BASIC_FIVE_STATUS_BONUS = 3  # 常规比赛属性加成
RACE_BASIC_PT_BONUS = 45  # G1比赛pt加成

# 每pt对应多少分
SCORE_PT_RATE_DEFAULT = 2.0
# 每级hint等价多少pt
HINT_LEVEL_PT_RATE_DEFAULT = 4

# 随机事件概率和强度
EVENT_PROB = 0.35
EVENT_STRENGTH_DEFAULT = 20

# 齿轮概率
MECHA_GEAR_PROB = 0.5
MECHA_GEAR_PROB_LINK_BONUS = 0.05

# 友人卡解锁出行概率
FRIEND_UNLOCK_PROB_LOW_JIBAN = 0.1  # 羁绊 < 60
FRIEND_UNLOCK_PROB_HIGH_JIBAN = 0.2  # 羁绊 >= 60

# 友人卡ID
FRIEND_CARD_YAYOI_SSR_ID = 30207  # SSR秋川
FRIEND_CARD_YAYOI_R_ID = 10109  # R秋川
FRIEND_CARD_LIANGHUA_SSR_ID = 30188  # SSR凉花
FRIEND_CARD_LIANGHUA_R_ID = 10104  # R凉花

# UGE目标等级
MECHA_TARGET_TOTAL_LEVEL = [600, 1000, 1400, 1900, 2400, 2400]

# 研究Lv提升量基础值 [是否合宿][无齿轮/齿轮/友情][主/副1/副2][人头数0-5]
MECHA_LV_GAIN_BASIC = [
    # 通常
    [
        # 通常
        [
            [7, 11, 14, 18, 21, 25],
            [2, 3, 4, 5, 6, 7],
            [1, 1, 2, 2, 3, 3],
        ],
        # 齿轮
        [
            [9, 13, 17, 21, 26, 30],
            [2, 4, 5, 6, 7, 8],
            [1, 1, 2, 3, 3, 4],
        ],
        # 友情
        [
            [0, 17, 21, 25, 29, 33],
            [0, 4, 6, 7, 8, 10],
            [0, 2, 2, 3, 4, 4],
        ],
    ],
    # 合宿
    [
        # 通常
        [
            [14, 18, 21, 25, 28, 32],
            [4, 5, 6, 7, 8, 9],
            [2, 2, 3, 3, 4, 4],
        ],
        # 齿轮
        [
            [17, 21, 26, 30, 34, 38],
            [5, 6, 7, 8, 10, 11],
            [2, 3, 3, 4, 4, 5],
        ],
        # 友情
        [
            [0, 25, 29, 33, 38, 42],
            [0, 7, 8, 10, 10, 12],
            [0, 3, 4, 4, 5, 5],
        ],
    ],
]

# 研究Lv提升量的副训练索引 [训练类型][主/副1/副2]
MECHA_LV_GAIN_SUB_TRAIN_IDX = [
    [0, 2, 1],  # 速度训练 -> 速(主), 力(副1), 耐(副2)
    [1, 4, 3],  # 耐力训练 -> 耐(主), 智(副1), 根(副2)
    [2, 1, 4],  # 力量训练 -> 力(主), 耐(副1), 智(副2)
    [3, 0, 2],  # 根性训练 -> 根(主), 速(副1), 力(副2)
    [4, 3, 0],  # 智力训练 -> 智(主), 根(副1), 速(副2)
]

# Link角色ID
MECHA_LINK_CHARAS = [1023, 1050, 1036, 1083, 1084]

# 评分模式
SCORING_NORMAL = 0  # 普通(评价点)模式
SCORING_RACE = 1  # 通用大赛模式
SCORING_MILE = 2  # 英里模式


# ============================================================================
# 神经网络输入输出维度
# ============================================================================

# 输入维度（参考UmaAi的NNInput.h和training/config.py）
# 全局信息 + 6张支援卡信息
NN_INPUT_C_GLOBAL = 587  # 全局信息通道数
NN_INPUT_C_CARD = 89  # 每张支援卡参数通道数
NN_INPUT_C_PERSON = 12  # 每个人头信息通道数（Dreams剧本暂不使用独立person）
NN_INPUT_C_CARDPERSON = NN_INPUT_C_CARD + NN_INPUT_C_PERSON  # 每个支援卡人头

NN_CARD_NUM = 6  # 支援卡数量
NN_HEAD_NUM = 0  # 场上人头数（Dreams剧本不使用独立person输入，0表示合并到card）

# 总输入维度
NN_INPUT_C = NN_INPUT_C_GLOBAL + NN_CARD_NUM * NN_INPUT_C_CARDPERSON + NN_HEAD_NUM * NN_INPUT_C_PERSON

# 输出维度
NN_OUTPUT_C_POLICY = 53  # 策略维度（标准动作数）
NN_OUTPUT_C_VALUE = 3  # 价值维度（平均分、标准差、乐观分）
NN_OUTPUT_C = NN_OUTPUT_C_POLICY + NN_OUTPUT_C_VALUE

# 兼容旧版变量名
Game_Input_C = NN_INPUT_C
Game_Input_C_Global = NN_INPUT_C_GLOBAL
Game_Input_C_Card = NN_INPUT_C_CARD
Game_Input_C_Person = NN_INPUT_C_PERSON
Game_Card_Num = NN_CARD_NUM
Game_Head_Num = NN_HEAD_NUM
Game_Output_C = NN_OUTPUT_C
Game_Output_C_Policy = NN_OUTPUT_C_POLICY
Game_Output_C_Value = NN_OUTPUT_C_VALUE


# ============================================================================
# 训练超参数
# ============================================================================

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

# 模型大小限制（376KB）
MODEL_SIZE_LIMIT = 376 * 1024

# 默认模型架构参数（EncoderMLPSimple）
DEFAULT_ENCODER_BLOCKS = 1
DEFAULT_ENCODER_FEATURES = 128
DEFAULT_MLP_BLOCKS = 3
DEFAULT_MLP_FEATURES = 256
DEFAULT_GLOBAL_FEATURES = 256


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


# ============================================================================
# 自我对弈参数
# ============================================================================

SELFPLAY_NUM_GAMES = 100  # 自我对弈局数
SELFPLAY_BATCH_SIZE = 16  # 自我对弈批大小
