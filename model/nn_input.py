"""
赛马娘AI训练框架 - NN输入编码

参考 UmaAi 的 NNInput.h 和 Game::getNNInputV1，将游戏状态编码为神经网络输入向量。
修复BUG-3：NN_INPUT_C_CARD对齐C++ 77维布局（29基础+35固有类型+13固有效果）
修复BUG-4：每个cardperson块中Person(12)在前、CardParam(77)在后，对齐C++布局
"""

import math
from typing import List
from simulator.game import Game
from simulator.action import TrainActionType
from simulator.person import PersonType, FriendStage
from simulator.bad_condition import BadConditionType
from simulator.scenarios.ramen import RamenScenario
from config import (
    NN_INPUT_C, NN_INPUT_C_GLOBAL, NN_INPUT_C_CARD, NN_INPUT_C_PERSON,
    NN_INPUT_C_CARDPERSON, NN_CARD_NUM, TOTAL_TURN,
    NN_INPUT_C_BC, NN_INPUT_C_RAMEN,
)

# C++ NNINPUT_CHANNELS_CARD_V1布局常量
_CARD_BASIC_C = 29      # 基础参数：类型one-hot(7)+友情(1)+干劲(1)+训练(1)+bonus(6)+智体(1)+hint(2)+预留(4)+hint概率(1)+得意(1)+失败率(1)+体力消费(1)+预留(2)
_CARD_UNIQUE_TYPE_C = 35 # 固有类型one-hot(0~34)
_CARD_UNIQUE_EFFECT_C = 13 # 固有效果值：速耐力根智pt(5)+友情(1)+干劲(1)+训练(1)+失败率(1)+体力消费(1)+智体(1)+预留(1)
assert _CARD_BASIC_C + _CARD_UNIQUE_TYPE_C + _CARD_UNIQUE_EFFECT_C == NN_INPUT_C_CARD


def encode_card_param(buf: List[float], ci: int, cp, game: 'Game') -> int:
    """编码支援卡参数到buf[ci:]，对齐C++ SupportCard::getCardParamNNInputV1
    
    77维布局：基础参数(29) + 固有类型one-hot(35) + 固有效果值(13)
    
    Args:
        buf: 输出缓冲区
        ci: 当前写入位置
        cp: SupportCard对象
        game: 游戏状态
    
    Returns:
        写入后的位置
    """
    # 先清零77维
    for i in range(NN_INPUT_C_CARD):
        buf[ci + i] = 0.0
    
    # ===== 基础参数 (29维, ci+0 ~ ci+28) =====
    # 0~6: cardType one-hot (7维)
    card_type = min(cp.card_type, 6)
    buf[ci + card_type] = 1.0
    
    # 7: 友情加成
    buf[ci + 7] = cp.you_qing_basic * 0.04
    # 8: 干劲加成
    buf[ci + 8] = cp.gan_jing_basic * 0.02
    # 9: 训练加成
    buf[ci + 9] = cp.xun_lian_basic * 0.05
    # 10~15: bonus (6维)
    for i in range(6):
        buf[ci + 10 + i] = cp.bonus_basic[i] * 0.5
    # 16: 智力彩圈体力
    buf[ci + 16] = cp.wiz_vital_bonus * 0.2
    # 17: hint等级 (>0时)
    if cp.hint_level > 0:
        buf[ci + 17] = cp.hint_level * 0.4
    # 18: 无hint标记 (hint_level==0时为1.0)
    else:
        buf[ci + 18] = 1.0
    # 19~22: 预留 (已清零)
    # 23: hint概率提升
    buf[ci + 23] = cp.hint_prob_increase * 0.02
    # 24: 得意率
    buf[ci + 24] = cp.de_yi_lv * 0.02
    # 25: 失败率下降
    buf[ci + 25] = cp.fail_rate_drop * 0.04
    # 26: 体力消费下降
    buf[ci + 26] = cp.vital_cost_drop * 0.05
    # 27~28: 预留 (已清零)
    
    # ===== 固有类型one-hot (35维, ci+29 ~ ci+63) =====
    # 固有效果类型编码（从SupportCard获取，默认0=无固有）
    unique_type = getattr(cp, 'unique_effect_type', 0)
    
    if unique_type == 0:
        buf[ci + 29 + 0] = 1.0  # 无固有
    elif unique_type in (1, 2):
        # 条件型固有，根据unique_param[1]区分80/100
        unique_param = getattr(cp, 'unique_effect_param', [0] * 6)
        if len(unique_param) > 1 and unique_param[1] == 80:
            buf[ci + 29 + 1] = 1.0
        elif len(unique_param) > 1 and unique_param[1] == 100:
            buf[ci + 29 + 2] = 1.0
        else:
            buf[ci + 29 + 1] = 1.0  # fallback
    elif 3 <= unique_type <= 14:
        buf[ci + 29 + unique_type] = 1.0
    elif unique_type == 16:
        # 购买技能型固有，简化编码
        unique_param = getattr(cp, 'unique_effect_param', [0] * 6)
        if len(unique_param) > 4 and unique_param[1] == 1 and unique_param[4] == 5:
            buf[ci + 29 + 16] = 1.0
        elif len(unique_param) > 4 and unique_param[1] == 1 and unique_param[4] == 3:
            buf[ci + 29 + 30] = 1.0
        elif len(unique_param) > 4 and unique_param[1] == 2 and unique_param[4] == 3:
            buf[ci + 29 + 31] = 1.0
        elif len(unique_param) > 4 and unique_param[1] == 3 and unique_param[4] == 3:
            buf[ci + 29 + 32] = 1.0
        else:
            buf[ci + 29 + 16] = 1.0  # fallback
    elif unique_type == 17:
        buf[ci + 29 + 17] = 1.0
    elif unique_type == 20:
        buf[ci + 29 + 20] = 1.0
    elif unique_type == 21:
        buf[ci + 29 + 21] = 1.0
    elif unique_type == 22:
        buf[ci + 29 + 22] = 1.0
    elif 6 <= unique_type <= 14:
        buf[ci + 29 + unique_type] = 1.0
    elif unique_type > 0 and unique_type <= 34:
        buf[ci + 29 + unique_type] = 1.0
    
    # ===== 固有效果值 (13维, ci+64 ~ ci+76) =====
    # 根据固有效果类型写入对应的效果值
    # 已清零，只写有值的
    if unique_type in (1, 2):
        unique_param = getattr(cp, 'unique_effect_param', [0] * 6)
        _write_unique_effect(buf, ci + _CARD_BASIC_C + _CARD_UNIQUE_TYPE_C, unique_param)
    
    return ci + NN_INPUT_C_CARD


def _write_unique_effect(buf: List[float], effect_base: int, params: List):
    """写入固有效果值，对齐C++ writeUniqueEffect
    
    效果值索引：
    0-4: 速耐力根智加成
    5: pt加成
    6: 友情加成
    7: 干劲加成
    8: 训练加成
    9: 失败率下降
    10: 体力消费下降
    11: 智力彩圈体力
    12: 预留
    """
    # 每对参数 (key, value) 从 params[2] 开始
    for idx in range(2, len(params) - 1, 2):
        key = params[idx]
        value = params[idx + 1]
        if key <= 0:
            continue
        elif key == 1:
            buf[effect_base + 6] = 0.04 * value
        elif key == 2:
            buf[effect_base + 7] = 0.02 * value
        elif key == 3:
            buf[effect_base + 0] = 0.5 * value
        elif key == 4:
            buf[effect_base + 1] = 0.5 * value
        elif key == 5:
            buf[effect_base + 2] = 0.5 * value
        elif key == 6:
            buf[effect_base + 3] = 0.5 * value
        elif key == 7:
            buf[effect_base + 4] = 0.5 * value
        elif key == 8:
            buf[effect_base + 8] = 0.05 * value
        elif key == 27:
            buf[effect_base + 9] = 0.04 * value
        elif key == 28:
            buf[effect_base + 10] = 0.05 * value
        elif key == 30:
            buf[effect_base + 5] = 0.5 * value
        elif key == 31:
            buf[effect_base + 11] = 0.2 * value
        elif key == 41:
            for i in range(5):
                buf[effect_base + i] = 0.5


def encode_person_info(buf: List[float], pi: int, p, game: 'Game', card_idx: int) -> int:
    """编码人头信息到buf[pi:]，对齐C++ Person::getCardNNInputV1
    
    12维布局（对齐C++）：羁绊/100 + 羁绊>=80 + 羁绊>=100 + 提示 + 预留3 + 训练位置one-hot5
    
    Args:
        buf: 输出缓冲区
        pi: 当前写入位置
        p: Person对象
        game: 游戏状态
        card_idx: 卡片索引
    
    Returns:
        写入后的位置
    """
    # 0: 羁绊/100
    buf[pi] = p.friendship / 100.0; pi += 1
    # 1: 羁绊>=80
    buf[pi] = 1.0 if p.friendship >= 80 else 0.0; pi += 1
    # 2: 羁绊>=100
    buf[pi] = 1.0 if p.friendship >= 100 else 0.0; pi += 1
    # 3: 是否有hint
    buf[pi] = 1.0 if p.is_hint else 0.0; pi += 1
    # 4~6: 预留（C++中为0）
    buf[pi] = 0.0; pi += 1
    buf[pi] = 0.0; pi += 1
    buf[pi] = 0.0; pi += 1
    # 7~11: 在哪个训练（one-hot 5维）
    for t in range(5):
        found = any(
            game.person_distribution[t][h] == card_idx
            for h in range(5)
        )
        buf[pi] = 1.0 if found else 0.0; pi += 1
    
    return pi


def encode_game_state(game: Game) -> List[float]:
    """将游戏状态编码为神经网络输入向量
    
    输入结构（对齐C++布局）：
    (全局信息156维)(卡槽1信息89维)...(卡槽6信息89维)
    
    全局信息156维 = 基础游戏状态132维 + BC8维 + Ramen16维
    每卡89维 = 人头信息12维(前) + 卡片参数77维(后)  ← BUG-4修复：对齐C++顺序
    
    参考 UmaAi 的 Game::getNNInputV1
    
    Args:
        game: 游戏状态
        
    Returns:
        长度为 NN_INPUT_C 的浮点数列表
    """
    buf = [0.0] * NN_INPUT_C
    idx = 0
    
    # ===== 基础游戏状态 (132维) =====
    # 回合数（归一化）
    buf[idx] = game.turn / TOTAL_TURN; idx += 1  # idx=1 after turn
    
    # 体力
    buf[idx] = game.vital / 120.0; idx += 1
    buf[idx] = game.max_vital / 120.0; idx += 1  # idx=3 after vital
    
    # 干劲 (one-hot 5维)
    for i in range(5):
        buf[idx] = 1.0 if game.motivation == i + 1 else 0.0; idx += 1  # idx=8 after motivation
    
    # 五维属性（归一化到0-1）
    for i in range(5):
        buf[idx] = game.five_status[i] / 2000.0; idx += 1  # idx=13 after five_status
    # 五维上限
    for i in range(5):
        buf[idx] = game.five_status_limit[i] / 2000.0; idx += 1  # idx=18 after five_status_limit
    # 成长率
    for i in range(5):
        buf[idx] = game.five_status_bonus[i] / 120.0; idx += 1  # idx=23 after five_status_bonus
    
    # 技能点
    buf[idx] = game.skill_pt / 1000.0; idx += 1
    # 技能分
    buf[idx] = game.skill_score / 5000.0; idx += 1  # idx=25 after skill_pt_score
    
    # 训练等级
    for i in range(5):
        buf[idx] = game.get_training_level(i) / 4.0; idx += 1  # idx=30 after training_level
    
    # 训练值 (5*6=30维)
    for t in range(5):
        for i in range(6):
            buf[idx] = game.train_value[t][i] / 100.0; idx += 1  # idx=60 after train_value
    
    # 体力变化
    for i in range(5):
        buf[idx] = game.train_vital_change[i] / 50.0; idx += 1  # idx=65 after train_vital_change
    
    # 失败率
    for i in range(5):
        buf[idx] = game.fail_rate[i] / 100.0; idx += 1  # idx=70 after fail_rate
    
    # 是否闪彩
    for i in range(5):
        buf[idx] = 1.0 if game.is_train_shining[i] else 0.0; idx += 1  # idx=75 after is_train_shining
    
    # 是否比赛回合
    buf[idx] = 1.0 if game.is_racing else 0.0; idx += 1  # idx=76 after is_racing
    
    # 是否夏合宿
    buf[idx] = 1.0 if game.is_xiahesu() else 0.0; idx += 1  # idx=77 after is_xiahesu
    
    # 是否可比赛
    buf[idx] = 1.0 if game.is_race_available() else 0.0; idx += 1  # idx=78 after is_race_available
    
    # 失败率bias
    buf[idx] = game.failure_rate_bias / 4.0; idx += 1  # idx=79 after failure_rate_bias
    
    # 特殊状态
    buf[idx] = 1.0 if game.is_qie_zhe else 0.0; idx += 1
    buf[idx] = 1.0 if game.is_ai_jiao else 0.0; idx += 1
    buf[idx] = 1.0 if game.is_positive_thinking else 0.0; idx += 1
    buf[idx] = 1.0 if game.is_refresh_mind else 0.0; idx += 1  # idx=83 after special_status
    
    # 友人卡状态
    buf[idx] = 1.0 if game.friend_type != 0 else 0.0; idx += 1
    buf[idx] = 1.0 if game.friend_is_ssr else 0.0; idx += 1
    buf[idx] = game.friend_stage / 3.0; idx += 1
    buf[idx] = game.friend_outgoing_used / 5.0; idx += 1  # idx=87 after friend_card
    
    # 非卡羁绊
    buf[idx] = game.friendship_noncard_yayoi / 100.0; idx += 1
    buf[idx] = game.friendship_noncard_reporter / 100.0; idx += 1  # idx=89 after noncard_friendship
    
    # 种马蓝因子
    for i in range(5):
        buf[idx] = game.zhong_ma_blue_count[i] / 6.0; idx += 1  # idx=94 after zhong_ma
    
    # Dreams剧本状态
    buf[idx] = game.mecha_en / 10.0; idx += 1
    buf[idx] = game.mecha_overdrive_energy / 6.0; idx += 1
    buf[idx] = 1.0 if game.mecha_overdrive_enabled else 0.0; idx += 1
    buf[idx] = 1.0 if game.mecha_any_lose else 0.0; idx += 1  # idx=98 after dreams_basic
    
    # 研究等级
    for i in range(5):
        buf[idx] = game.mecha_rival_lv[i] / 700.0; idx += 1
    buf[idx] = game.mecha_rival_lv_total / 3500.0; idx += 1  # idx=104 after rival_lv
    
    # 机甲升级
    for i in range(3):
        for j in range(3):
            buf[idx] = game.mecha_upgrade[i][j] / 5.0; idx += 1
    for i in range(3):
        buf[idx] = game.mecha_upgrade_total[i] / 15.0; idx += 1  # idx=116 after mecha_upgrade
    
    # 齿轮状态
    for i in range(5):
        buf[idx] = 1.0 if game.mecha_has_gear[i] else 0.0; idx += 1  # idx=121 after mecha_gear
    
    # UGE胜负
    for i in range(5):
        buf[idx] = game.mecha_win_history[i] / 2.0; idx += 1  # idx=126 after mecha_win
    
    # 训练倍率
    for i in range(6):
        buf[idx] = game.mecha_training_status_multiplier[i] - 1.0; idx += 1  # idx=132 after training_multiplier

    # 验证基础游戏状态维度
    _GLOBAL_BASE_DIM = 132
    assert idx == _GLOBAL_BASE_DIM, f"基础全局编码维度不匹配: 实际{idx}, 期望{_GLOBAL_BASE_DIM}"

    # ===== バッドコンディション状态 (8维) =====
    # 6种状态one-hot
    for bc_type in [BadConditionType.BAD, BadConditionType.LAZY,
                    BadConditionType.FAT, BadConditionType.HEADACHE,
                    BadConditionType.SKIN, BadConditionType.LATE_BED]:
        buf[idx] = 1.0 if game.bc_manager.has(bc_type) else 0.0; idx += 1
    # BC数量（归一化）
    buf[idx] = game.bc_manager.count / 6.0; idx += 1
    # 是否可治愈（保健室/お休み）
    has_healable = (game.bc_manager.count > 0)
    buf[idx] = 1.0 if has_healable else 0.0; idx += 1  # idx=140 after bad_condition

    assert idx == _GLOBAL_BASE_DIM + NN_INPUT_C_BC, \
        f"BC编码维度不匹配: 实际{idx}, 期望{_GLOBAL_BASE_DIM + NN_INPUT_C_BC}"

    # ===== Ramen剧本状态 (16维) =====
    ramen = game.scenario if isinstance(game.scenario, RamenScenario) else None
    if ramen is not None:
        # 隠し味の秘訣数量（归一化到~50）
        buf[idx] = ramen.kakushimi_count / 50.0; idx += 1
        # 試食会次数
        buf[idx] = ramen.tasting_count / 5.0; idx += 1
        # コツ等级（5维）
        for i in range(5):
            buf[idx] = ramen.kotsu_level[i] / 3.0; idx += 1
        # CheckPoint进度
        if ramen.expected_checkpoint_pt > 0:
            buf[idx] = ramen.checkpoint_pt / ramen.expected_checkpoint_pt; idx += 1
        else:
            buf[idx] = 0.0; idx += 1
        # 地域贡献（5个区域，归一化）
        for region_id in range(5):
            buf[idx] = ramen.region_points.get(region_id, 0) / 100.0; idx += 1
        # SpecialFeelingNum
        buf[idx] = ramen.special_feeling_num / 10.0; idx += 1
        # Feeling值（2维：取前2个feeling_type的值）
        feeling_vals = list(ramen.feeling_values.values())
        buf[idx] = feeling_vals[0] / 100.0 if len(feeling_vals) > 0 else 0.0; idx += 1
        buf[idx] = feeling_vals[1] / 100.0 if len(feeling_vals) > 1 else 0.0; idx += 1
    else:
        # 非Ramen剧本：16维全0
        for _ in range(NN_INPUT_C_RAMEN):
            buf[idx] = 0.0; idx += 1  # idx=156 after ramen

    # 验证全局段总维度（基础+BC+Ramen，无零填充）
    assert idx == NN_INPUT_C_GLOBAL, \
        f"全局编码维度不匹配: 实际{idx}, 配置{NN_INPUT_C_GLOBAL}"

    # ===== 支援卡信息 =====
    # BUG-4修复：对齐C++布局，每个cardperson块 = [person_info(12)][card_param(77)]
    for card_idx in range(NN_CARD_NUM):
        card_start = NN_INPUT_C_GLOBAL + card_idx * NN_INPUT_C_CARDPERSON
        
        if card_idx < len(game.persons):
            p = game.persons[card_idx]
            cp = p.card_param
            
            # BUG-4修复：Person在前(12维)，CardParam在后(77维)
            # 人头信息（NN_INPUT_C_PERSON=12维）
            pi = card_start
            pi = encode_person_info(buf, pi, p, game, card_idx)
            
            # 验证人头信息维度
            assert pi - card_start == NN_INPUT_C_PERSON, \
                f"卡片{card_idx}人头维度不匹配: 实际{pi-card_start}, 配置{NN_INPUT_C_PERSON}"
            
            # 支援卡参数（NN_INPUT_C_CARD=77维）
            ci = card_start + NN_INPUT_C_PERSON
            ci = encode_card_param(buf, ci, cp, game)

            # 验证卡片参数维度
            assert ci - card_start == NN_INPUT_C_CARDPERSON, \
                f"卡片{card_idx}卡槽维度不匹配: 实际{ci-card_start}, 配置{NN_INPUT_C_CARDPERSON}"

    # 更新idx到卡片编码完成后的位置
    idx = NN_INPUT_C_GLOBAL + NN_CARD_NUM * NN_INPUT_C_CARDPERSON

    # 最终断言确保编码维度与NN_INPUT_C严格对齐
    assert idx == NN_INPUT_C, \
        f"编码维度不匹配: 实际idx={idx}, 配置NN_INPUT_C={NN_INPUT_C}"

    return buf
