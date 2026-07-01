"""
赛马娘AI训练框架 - 神经网络维度与模型超参配置

包含NN输入输出维度定义和模型架构参数。
"""

# ============================================================================
# 神经网络输入输出维度
# ============================================================================

# 输入维度（参考UmaAi的NNInput.h和training/config.py）
# 全局信息 + 6张支援卡信息（每卡含人头信息+卡片参数）
# 修复BUG-3/BUG-4：对齐C++ NNINPUT_CHANNELS_CARD_V1=77布局和Person/Card顺序
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

# BUG-3修复：对齐C++ NNINPUT_CHANNELS_CARD_V1=77
# 77维布局：基础参数29(类型one-hot7+友情1+干劲1+训练1+bonus6+智体1+hint2+预留4+hint概率1+得意1+失败率1+体力消费1+预留2)
#          +固有类型one-hot35(类型0~34)
#          +固有效果值13(速耐力根智pt5+友情1+干劲1+训练1+失败率1+体力消费1+智体1)
NN_INPUT_C_CARD = 77    # 每张支援卡参数维度（对齐C++ NNINPUT_CHANNELS_CARD_V1=77：29基础+35固有类型+13固有效果）

# BUG-4修复：对齐C++ Person/Card顺序 [person_info(12)][card_param(77)]
# C++ Person布局(12维)：羁绊1+羁绊>=80 1+羁绊>=100 1+提示1+预留3+训练位置one-hot5
NN_INPUT_C_PERSON = 12  # 每个人头信息维度（对齐C++：羁绊1+羁绊阈值2+提示1+预留3+训练位置5）
NN_INPUT_C_CARDPERSON = NN_INPUT_C_PERSON + NN_INPUT_C_CARD  # 每个卡槽维度（人头+卡片，89维）

NN_CARD_NUM = 6  # 支援卡数量
NN_HEAD_NUM = 0  # 场上人头数（Dreams剧本不使用独立person输入，0表示合并到card）

# 总输入维度（156 + 6*89 = 690）
NN_INPUT_C = NN_INPUT_C_GLOBAL + NN_CARD_NUM * NN_INPUT_C_CARDPERSON + NN_HEAD_NUM * NN_INPUT_C_PERSON

# 输出维度
# BUG-2修复：NN_OUTPUT_C_POLICY=53→50，对齐C++ Dreams剧本配置
NN_OUTPUT_C_POLICY = 50  # 策略维度（对齐C++ NNOUTPUT_CHANNELS_POLICY_V1=50）
NN_OUTPUT_C_VALUE = 3  # 价值维度（平均分、标准差、乐观分）
NN_OUTPUT_C = NN_OUTPUT_C_POLICY + NN_OUTPUT_C_VALUE

# ============================================================================
# 模型架构参数
# ============================================================================

# 模型大小限制（376KB）
MODEL_SIZE_LIMIT = 376 * 1024

# 默认模型架构参数（EncoderMLPSimple）
DEFAULT_ENCODER_BLOCKS = 1
DEFAULT_ENCODER_FEATURES = 128
DEFAULT_MLP_BLOCKS = 3
DEFAULT_MLP_FEATURES = 256
DEFAULT_GLOBAL_FEATURES = 256
