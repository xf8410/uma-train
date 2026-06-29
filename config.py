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

# ============================================================================
# バッドコンディション常量
# ============================================================================

# 获取概率（每回合，从实测统计）
BC_ACQUIRE_PROB_BAD = 0.03          # 練習ベタ
BC_ACQUIRE_PROB_LAZY = 0.02        # なまけ癖
BC_ACQUIRE_PROB_FAT = 0.015        # 太り気味
BC_ACQUIRE_PROB_HEADACHE = 0.02    # 片頭痛
BC_ACQUIRE_PROB_SKIN = 0.01        # 肌荒れ
BC_ACQUIRE_PROB_LATE_BED = 0.03    # 夜ふかし

# 治愈概率
BC_HEAL_REST_LATE_BED = 0.5        # お休み治愈夜ふかし概率
BC_HEAL_REST_SKIN = 0.4            # お休み治愈肌荒れ概率

# なまけ癖触发概率和冷却
BC_LAZY_TRIGGER_PROB = 0.4         # なまけ癖触发训练跳过概率
BC_LAZY_COOLDOWN = 3               # 触发后冷却回合数
BC_LAZY_INITIAL_COOLDOWN = 2       # 获取后初始冷却

# 肌荒れやる気下降概率
BC_SKIN_MOTIVATION_DRAIN_PROB = 0.35

# やる気下降事件不重复间隔（2.5周年后）
MOTIVATION_DOWN_COOLDOWN = 5

# 保健室：2.5周年后必消1个（确定性）
CLINIC_HEAL_COUNT = 1

# やる気倍率表（1=絶不調 ... 5=絶好調）
MOTIVATION_TRAIN_MULT = {1: 0.8, 2: 0.9, 3: 1.0, 4: 1.1, 5: 1.2}

# お休み基础回复量（按やる気）
MOTIVATION_REST_VITAL_BASE = {1: 50, 2: 55, 3: 60, 4: 65, 5: 70}

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
# 人员ID常量（原game.py模块级常量）
# ============================================================================

PSID_NONE = -1        # 未分配
PSID_NONCARD_YAYOI = 6  # 非卡理事长
PSID_NONCARD_REPORTER = 7  # 非卡记者
PSID_NPC = 8          # NPC

# 友人卡类型
FRIEND_TYPE_NONE = 0
FRIEND_TYPE_LIANGHUA = 1  # 凉花
FRIEND_TYPE_YAYOI = 2     # 理事长(秋川)

# ============================================================================
# 魔法数字提取（含中文注释说明来源和含义）
# ============================================================================

# 属性2倍率阈值：属性值>1200后，每2点只算1点（游戏机制）
PROPERTY_DOUBLE_THRESHOLD = 1200

# 失败率公式分母（来源：反编译确认的CY公式）
FAIL_RATE_FORMULA_DENOM = 40.0

# URA结算技能分（第三年URA结算固定+170技能分）
URA_SKILL_SCORE = 170

# 初始技能分公式系数（3星以上: 170*(stars-2), 以下: 120*stars）
INITIAL_SKILL_SCORE_HIGH = 170
INITIAL_SKILL_SCORE_LOW = 120

# 评分：属性>100部分的倍率（来源：评价点计算公式）
SCORE_ABOVE_100_MULT = 1.1

# 评分：属性权重 [速,耐,力,根,智]（用于_final_score_sum）
SCORING_WEIGHTS = [5, 3, 3, 3, 3]

# 属性上限超过1200后的折半系数（来源：游戏机制，>1200后每2点算1点）
PROPERTY_HALVE_THRESHOLD = 1200

# 非卡理事长出现概率权重（5个训练各100，不出现200）
NONCARD_YAYOI_WEIGHTS = [100, 100, 100, 100, 100, 200]

# NPC出现概率权重（5个训练各100，不出现100）
NPC_DISTRIBUTION_WEIGHTS = [100, 100, 100, 100, 100, 100]

# 随机事件概率
RANDOM_EVENT_PROB = 0.35  # 支援卡连续事件触发概率
MOTIVATION_DOWN_PROB = 0.04  # やる気下降事件概率
CHARA_EVENT_PROB = 0.1  # 马娘随机事件概率
VITAL_EVENT_SMALL_PROB = 0.10  # 小体力事件概率(+5)
VITAL_EVENT_BIG_PROB = 0.02  # 大体力事件概率(+30)
MOTIVATION_UP_EVENT_PROB = 0.02  # やる気上升事件概率

# 人头hint基础概率
HINT_BASE_PROB = 0.06

# overdrive拉人防死循环上限
OVERDRIVE_INVITE_MAX_TRY = 1000

# 种马蓝因子属性上限加成系数
ZHONGMA_BLUE_LIMIT_FACTOR = 5.34
ZHONGMA_BLUE_LIMIT_MULT = 2  # 种马蓝因子上限倍率

# Dreams研究等级上限（按回合区间）
MECHA_RIVAL_LV_LIMITS = [(24, 200), (36, 300), (48, 400), (60, 500), (72, 600), (999, 700)]

# Dreams齿轮加成（按回合区间，来源：反编译）
MECHA_GEAR_BONUS_TABLE = [(12, 3), (24, 6), (36, 10), (48, 16), (60, 20), (72, 25), (999, 30)]

# overdrive倍率
OVERDRIVE_TRAIN_MULT = 1.25

# overdrive EN消耗
OVERDRIVE_EN_COST = 3

# URA回合开始
URA_START_TURN = 72

# 研究等级提升量bonus表（来源：反编译）
MECHA_LV_GAIN_BONUS_TABLE = {5: 40, 4: 33, 3: 26, 2: 18, 1: 10}



# ============================================================================
# 神经网络输入输出维度
# ============================================================================

# 输入维度（参考UmaAi的NNInput.h和training/config.py）
# 全局信息 + 6张支援卡信息（每卡含卡片参数+人头信息）
# 修复P0-7：去除零填充预留空间，维度与实际编码严格对齐
#
# 全局信息维度明细（共156维）：
#   基础游戏状态132维：回合1+体力2+干劲5+属性5+上限5+成长率5+技能点1+技能分1
#     +训练等级5+训练值30+体力变化5+失败率5+闪彩5+比赛标志3+失败率bias1
#     +特殊状态4+友人卡4+非卡羁绊2+种马5+Dreams状态4+研究等级6+机甲升级12
#     +齿轮5+UGE胜负5+训练倍率6
#   バッドコンディション 8维：6种状态one-hot + 数量 + 治愈可能
#   Ramen剧本 16维：隠し味1+試食会1+コツ5+CP1+地域5+Feeling数1+Feeling值2
NN_INPUT_C_GLOBAL = 156  # 全局信息维度（基础132+BC8+Ramen16，无预留零填充）
NN_INPUT_C_BC = 8       # バッドコンディション维度（6种状态+count+治愈可能）
NN_INPUT_C_RAMEN = 16   # Ramen剧本维度（隠し味1+試食会1+コツ5+CP1+地域5+Feeling数1+Feeling值2）
NN_INPUT_C_CARD = 26    # 每张支援卡参数维度（实际编码：类型1+友情1+感性1+训练1+bonus6+智力体力1+初始6+提示2+得意1+失败率1+体力消减1+link1+赛後1+回复1+效果1）
NN_INPUT_C_PERSON = 12  # 每个人头信息维度（羁绊1+提示1+类型1+记录1+训练位置5+闪彩2+友人1）
NN_INPUT_C_CARDPERSON = NN_INPUT_C_CARD + NN_INPUT_C_PERSON  # 每个卡槽维度（卡片+人头，38维）

NN_CARD_NUM = 6  # 支援卡数量
NN_HEAD_NUM = 0  # 场上人头数（Dreams剧本不使用独立person输入，0表示合并到card）

# 总输入维度（156 + 6*38 = 384）
NN_INPUT_C = NN_INPUT_C_GLOBAL + NN_CARD_NUM * NN_INPUT_C_CARDPERSON + NN_HEAD_NUM * NN_INPUT_C_PERSON

# 输出维度
NN_OUTPUT_C_POLICY = 53  # 策略维度（标准动作数）
NN_OUTPUT_C_VALUE = 3  # 价值维度（平均分、标准差、乐观分）
NN_OUTPUT_C = NN_OUTPUT_C_POLICY + NN_OUTPUT_C_VALUE

# 兼容旧版变量名
Game_Input_C = NN_INPUT_C
Game_Input_C_Global = NN_INPUT_C_GLOBAL
Game_Input_C_Card = NN_INPUT_C_CARD
Game_Input_C_CardPerson = NN_INPUT_C_CARDPERSON  # 卡槽维度（卡片+人头，用于network输入层）
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
# 自我对弈参数
# ============================================================================

SELFPLAY_NUM_GAMES = 100  # 自我对弈局数
SELFPLAY_BATCH_SIZE = 16  # 自我对弈批大小


# ============================================================================
# MCTS决策归因日志参数
# ============================================================================

# 手写评估默认标准差（用于非NN评估时的stdev估算）
# 原来硬编码在_evaluate_game和selfplay里，现统一到config
HANDWRITTEN_STDEV_BASE = 500.0    # 手写评估初始标准差
HANDWRITTEN_STDEV_FLOOR = 100.0   # 手写评估最终标准差（游戏接近结束时）

# NN vs 手写对账模式（默认关闭，仅诊断用）
COMPARE_WITH_HANDWRITTEN = False


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
