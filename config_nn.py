"""
赛马娘AI训练框架 - 神经网络维度与模型超参配置

包含NN输入输出维度定义和模型架构参数。
"""

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
