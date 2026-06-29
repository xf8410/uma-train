"""
赛马娘AI训练框架 - 训练计算模块

从game.py提取的训练相关计算函数。
包含训练值计算、失败率计算、属性折算等。
"""

import math
from typing import List

from .action import TrainActionType
from .person import PersonType
from config import (
    FAIL_RATE_BASIC, TRAINING_BASIC_VALUE, MAX_PERSON_PER_TRAIN,
    MECHA_LV_GAIN_BASIC, MECHA_LV_GAIN_SUB_TRAIN_IDX,
    MECHA_TARGET_TOTAL_LEVEL,
    # 新增命名常量
    PROPERTY_DOUBLE_THRESHOLD, FAIL_RATE_FORMULA_DENOM,
    HINT_BASE_PROB, MECHA_GEAR_BONUS_TABLE, MECHA_LV_GAIN_BONUS_TABLE,
    OVERDRIVE_TRAIN_MULT, OVERDRIVE_EN_COST,
    MECHA_RIVAL_LV_LIMITS,
)


# P2-7修复：训练值上下限的上限截断值（来源：游戏引擎训练值下限/上限，反编译确认单次训练单项属性增益不超过此值）
TRAIN_VALUE_LOWER_CAP = 100
TRAIN_VALUE_UPPER_CAP = 100


def calc_real_status_gain(game, value: int, gain: int) -> int:
    """考虑1200以上为2的倍数的实际属性增加值
    
    属性2倍率阈值：属性值>1200后，每2点只算1点（游戏机制）
    """
    new_value = value + gain
    if new_value <= PROPERTY_DOUBLE_THRESHOLD:
        return gain
    if gain == 1:
        return 2
    return (new_value // 2) * 2 - value


def calc_failure_rate(game, train_type: int, fail_rate_multiply: float = 1.0) -> int:
    """计算训练失败率
    
    失败率公式分母（来源：反编译确认的CY公式）
    
    Args:
        game: Game实例
        train_type: 训练类型(0-4)
        fail_rate_multiply: 失败率乘数
        
    Returns:
        失败率百分比(0-100)
    """
    tlevel = game.get_training_level(train_type)
    x0 = 0.1 * FAIL_RATE_BASIC[train_type][tlevel]
    
    f = 0.0
    if game.vital < x0:
        f = (100 - game.vital) * (x0 - game.vital) / FAIL_RATE_FORMULA_DENOM
    
    f = max(0, min(f, 99))  # 无练习上手最高99%
    f *= fail_rate_multiply
    
    fr = math.ceil(f) + game.failure_rate_bias
    return max(0, min(fr, 100))


def calc_training_value_single(game, tra: int):
    """计算单个训练的数值
    
    参考 Game.cpp 的 calculateTrainingValueSingle
    """
    head_num = 0      # 人头数
    shining_num = 0   # 闪彩数

    tlevel = game.get_training_level(tra)

    is_card_shining_record = [False] * 6

    # 统计人头和闪彩
    for h in range(MAX_PERSON_PER_TRAIN):
        p_idx = game.person_distribution[tra][h]
        if p_idx < 0:
            break
        if p_idx == 8:  # PSID_NPC
            head_num += 1
            continue
        if p_idx >= 6:
            continue  # 非支援卡

        head_num += 1
        p = game.persons[p_idx]
        if game.is_card_shining(p_idx, tra):
            shining_num += 1
            is_card_shining_record[p_idx] = True

    game.is_train_shining[tra] = shining_num > 0

    # 基础值
    basic_value = list(TRAINING_BASIC_VALUE[tra][tlevel])
    vital_cost_basic = -basic_value[6]  # 体力消耗（取正值）

    # 累计支援卡效果
    total_xun_lian = 0
    total_gan_jing = 0
    total_you_qing_multiplier = 1.0
    vital_cost_multiplier = 1.0
    fail_rate_multiplier = 1.0

    for h in range(MAX_PERSON_PER_TRAIN):
        pid = game.person_distribution[tra][h]
        if pid < 0:
            break
        if pid >= 6:
            continue

        p = game.persons[pid]
        is_this_shining = is_card_shining_record[pid]
        
        eff = p.card_param.get_card_effect(
            is_this_shining, tra, p.friendship, p.card_record,
            head_num, shining_num
        )

        # 基础值bonus
        for i in range(6):
            if basic_value[i] > 0:
                basic_value[i] += int(eff.bonus[i])

        # 闪彩效果
        if is_card_shining_record[pid]:
            total_you_qing_multiplier *= (1 + 0.01 * eff.you_qing)
            if tra == TrainActionType.WIT:
                vital_cost_basic -= eff.vital_bonus

        total_xun_lian += eff.xun_lian
        total_gan_jing += eff.gan_jing
        vital_cost_multiplier *= (1 - 0.01 * eff.vital_cost_drop)
        fail_rate_multiplier *= (1 - 0.01 * eff.fail_rate_drop)

    # overdrive体力消耗减半
    if game.mecha_overdrive_enabled and game.mecha_upgrade_total[0] >= 15:
        vital_cost_multiplier *= 0.5

    # 体力变化
    vital_change_int = -int(vital_cost_basic * vital_cost_multiplier) if vital_cost_basic > 0 else -vital_cost_basic
    vital_change_int = max(-game.vital, min(vital_change_int, game.max_vital - game.vital))
    game.train_vital_change[tra] = vital_change_int
    game.fail_rate[tra] = game.calculate_failure_rate(tra, fail_rate_multiplier)

    # 支援卡乘区
    card_multiplier = (
        (1 + 0.05 * head_num) *
        (1 + 0.01 * total_xun_lian) *
        (1 + 0.1 * (game.motivation - 3) * (1 + 0.01 * total_gan_jing)) *
        total_you_qing_multiplier
    )

    # 下层值
    for i in range(6):
        is_related = basic_value[i] != 0
        bvl = basic_value[i]
        uma_bonus = 1 + 0.01 * game.five_status_bonus[i] if i < 5 else 1
        game.train_value_lower[tra][i] = bvl * card_multiplier * uma_bonus

    # 彩圈必有齿轮
    if shining_num > 0:
        game.mecha_has_gear[tra] = True

    # 上层（剧本乘区）
    scenario_train_multiplier = 1.0

    # 研究等级加成
    lv_bonus = (6 + 0.06 * game.mecha_rival_lv[tra]) if game.mecha_rival_lv[tra] > 1 else 0
    if game.mecha_linkeffect_lvbonus:
        lv_bonus *= 1.5
    scenario_train_multiplier *= (1 + 0.01 * lv_bonus)

    # 齿轮加成（按回合区间，来源：反编译）
    if game.mecha_has_gear[tra]:
        gear_bonus = 0
        for threshold, bonus in MECHA_GEAR_BONUS_TABLE:
            if game.turn < threshold:
                gear_bonus = bonus
                break
        scenario_train_multiplier *= (1 + 0.01 * gear_bonus)

    # 胸3号友情加成
    if shining_num > 0:
        friendship_bonus = 2 * game.mecha_upgrade[1][2]
        scenario_train_multiplier *= (1 + 0.01 * friendship_bonus)

    # overdrive加成
    if game.mecha_overdrive_enabled:
        scenario_train_multiplier *= OVERDRIVE_TRAIN_MULT
        head_bonus = 1 if game.mecha_upgrade_total[1] >= 3 else (3 if game.mecha_upgrade_total[1] >= 12 else 0)
        scenario_train_multiplier *= (1 + 0.01 * head_num * head_bonus)

    # 计算最终值
    for i in range(6):
        lower = game.train_value_lower[tra][i]
        lower = min(lower, TRAIN_VALUE_LOWER_CAP)  # 训练值下限截断（游戏引擎上限）
        game.train_value_lower[tra][i] = lower
        
        total = int(lower * scenario_train_multiplier * game.mecha_training_status_multiplier[i])
        upper = total - lower
        upper = min(upper, TRAIN_VALUE_UPPER_CAP)  # 训练值上限截断（游戏引擎上限）
        
        if i < 5:
            lower = game.calculate_real_status_gain(game.five_status[i], lower)
            upper = game.calculate_real_status_gain(game.five_status[i] + lower, upper)
        
        game.train_value[tra][i] = upper + lower

    # 研究等级提升量
    calc_lv_gain_single(game, tra, head_num, shining_num > 0)

    # 剧本修改训练值（如Ramen的隠し味/コツ加成）
    if game.scenario is not None:
        game.train_value[tra] = game.scenario.modify_training_value(game, tra, list(game.train_value[tra]))


def calc_lv_gain_single(game, tra: int, head_num: int, is_shining: bool):
    """计算每个训练加多少研究等级"""
    xhs = game.is_xiahesu()
    group = 0 if not game.mecha_has_gear[tra] else (1 if not is_shining else 2)
    
    for i in range(5):
        game.mecha_lv_gain[tra][i] = 0
    
    for sub in range(3):
        train_type = MECHA_LV_GAIN_SUB_TRAIN_IDX[tra][sub]
        basic = MECHA_LV_GAIN_BASIC[int(xhs)][group][sub][head_num]
        multiplier = game.mecha_lv_gain_multiplier[train_type]
        gain = int(multiplier * basic)
        if gain == basic and multiplier > 1:
            gain += 1
        game.mecha_lv_gain[tra][train_type] = gain


def calc_training_value(game):
    """计算所有训练分别加多少"""
    # 重新计算统计信息
    game.mecha_rival_lv_total = sum(game.mecha_rival_lv)
    
    # 研究等级上限（按回合区间）
    for threshold, limit in MECHA_RIVAL_LV_LIMITS:
        if game.turn < threshold:
            game.mecha_rival_lv_limit = limit
            break

    for i in range(3):
        game.mecha_upgrade_total[i] = sum(game.mecha_upgrade[i])

    # 属性加成倍率
    for i in range(5):
        m = 1.0
        if game.mecha_overdrive_enabled:
            upgrade_group = (
                game.mecha_upgrade_total[2] if i in (0, 2) else
                game.mecha_upgrade_total[1] if i in (1, 3) else
                game.mecha_upgrade_total[0]
            )
            if upgrade_group >= 9:
                count = 1 + (game.mecha_rival_lv_total - 1) // 200
                m *= (1 + 0.03 * count)
            elif upgrade_group >= 6:
                count = 1 + (game.mecha_rival_lv_total - 1) // 300
                m *= (1 + 0.03 * count)
        game.mecha_training_status_multiplier[i] = m

    # pt倍率
    ptb = 1.0 * (1 + game.mecha_upgrade[2][2] * 0.12)
    if game.mecha_overdrive_enabled and game.mecha_upgrade_total[2] >= 15:
        count = 1 + (game.mecha_rival_lv_total - 1) // 150
        ptb *= (1 + 0.03 * count)
    game.mecha_training_status_multiplier[5] = ptb

    # 研究等级提升量倍率（来源：反编译）
    for i in range(5):
        upgrade_lv = (
            game.mecha_upgrade[2][0] if i == 0 else
            game.mecha_upgrade[1][0] if i == 1 else
            game.mecha_upgrade[2][1] if i == 2 else
            game.mecha_upgrade[1][1] if i == 3 else
            game.mecha_upgrade[0][0]
        )
        lv_gain_bonus = MECHA_LV_GAIN_BONUS_TABLE.get(upgrade_lv, 0)
        if game.mecha_overdrive_enabled:
            if game.mecha_upgrade_total[0] >= 12:
                lv_gain_bonus += 25
            elif game.mecha_upgrade_total[0] >= 9:
                lv_gain_bonus += 20
            elif game.mecha_upgrade_total[0] >= 6:
                lv_gain_bonus += 15
        game.mecha_lv_gain_multiplier[i] = 1.0 + 0.01 * lv_gain_bonus

    for i in range(5):
        game.calculate_training_value_single(i)
