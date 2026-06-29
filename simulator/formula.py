"""
赛马娘AI训练框架 - 公式层

硬逻辑保证层：训练失败率、やる気倍率、体力消耗等核心公式。
纯数据驱动一定出错，必须有公式层硬逻辑保证。

参考：
  - UmaAi Game.cpp 中的训练计算逻辑
  - 2.5周年平衡补丁(2025.11.11)
  - 实测统计的失败率映射表
"""

import math
from typing import List, Optional, Tuple

from .bad_condition import BadConditionManager, BadConditionType
from config import (
    FAIL_RATE_BASIC, TRAINING_BASIC_VALUE,
    TOTAL_TURN, BASIC_FIVE_STATUS_LIMIT,
)


# ============================================================================
# やる気倍率表
# ============================================================================

# やる気(1-5) → 训练值倍率
# 1=絶不調, 2=不調, 3=普通, 4=好調, 5=絶好調
MOTIVATION_MULTIPLIER = {
    1: 0.8,   # 絶不調
    2: 0.9,   # 不調
    3: 1.0,   # 普通
    4: 1.1,   # 好調
    5: 1.2,   # 絶好調
}

# やる気 → 体力回复量（お休み基础）
MOTIVATION_REST_VITAL = {
    1: 50,   # 絶不調
    2: 55,   # 不調
    3: 60,   # 普通
    4: 65,   # 好調
    5: 70,   # 絶好調
}


# ============================================================================
# 训练失败率公式
# ============================================================================

def calc_fail_rate(
    vital: int,
    motivation: int,
    train_idx: int,
    train_level: int,
    bc_manager: BadConditionManager,
    failure_rate_bias: int = 0,
    is_race_turn: bool = False,
) -> int:
    """计算训练失败率（0-100百分比）

    核心公式（参考UmaAi Game.cpp）：
      x0 = 0.1 * FAIL_RATE_BASIC[train][level]
      当 vital < x0 时:
        f = (100 - vital) * (x0 - vital) / 40.0
      否则:
        f = 0

    修正项：
      - やる気修正：絶不調/不調时失败率上升
      - バッドコンディション修正：練習ベタ +2%
      - failure_rate_bias：其他加成
      - 2.5周年补丁：体力上升时失败率降低（已在基础参数中体现）

    Args:
        vital: 当前体力(0-100)
        motivation: やる気(1-5)
        train_idx: 训练类型(0-4)
        train_level: 训练等级(0-4)
        bc_manager: バッドコンディション管理器
        failure_rate_bias: 额外失败率偏移
        is_race_turn: 是否比赛回合

    Returns:
        失败率百分比(0-100)
    """
    # 比赛回合无失败率
    if is_race_turn:
        return 0

    # 智力训练无失败
    if train_idx == 4:  # WIT
        return 0

    # 基础失败率计算
    fail_basic = FAIL_RATE_BASIC[train_idx][train_level]
    x0 = 0.1 * fail_basic

    if vital >= x0:
        base_rate = 0.0
    else:
        base_rate = (100.0 - vital) * (x0 - vital) / 40.0

    # やる気修正
    motivation_mod = 0.0
    if motivation <= 1:  # 絶不調
        motivation_mod = 5.0
    elif motivation == 2:  # 不調
        motivation_mod = 2.0

    # バッドコンディション修正
    bc_mod = float(bc_manager.get_fail_rate_bonus())

    # 合计
    total = base_rate + motivation_mod + bc_mod + failure_rate_bias

    # 钳制到0-100
    return max(0, min(100, int(total)))


def calc_fail_rate_from_condition_set(
    condition_set_id: int,
    vital: int,
    motivation: int,
) -> int:
    """从condition_set_id计算失败率（备用方案）

    condition_set_id 507-548 对应不同训练类型和等级。
    当无法使用训练等级信息时，直接从condition_set_id推算。

    Args:
        condition_set_id: MasterDB中的condition_set_id
        vital: 当前体力
        motivation: やる気

    Returns:
        失败率百分比
    """
    # condition_set_id → (train_idx, level) 映射
    # 507-511: 耐力Lv0-4, 516-520: 力量Lv0-4, 520-536: 速度Lv0-4, 532-548: 根性Lv0-4
    csid_map = {}
    for lv in range(5):
        csid_map[507 + lv * 4] = (1, lv)   # 耐力
        csid_map[516 + lv * 4] = (2, lv)   # 力量
        csid_map[520 + lv * 4] = (3, lv)   # 速度
        csid_map[532 + lv * 4] = (4, lv)   # 根性

    if condition_set_id in csid_map:
        train_idx, level = csid_map[condition_set_id]
    else:
        # fallback: 从id范围估算
        train_idx = min(4, max(0, (condition_set_id - 507) // 8))
        level = min(4, max(0, (condition_set_id - 507) % 5))

    x0 = 0.1 * FAIL_RATE_BASIC[train_idx][level]
    if vital >= x0:
        return 0
    return max(0, min(100, int((100.0 - vital) * (x0 - vital) / 40.0)))


# ============================================================================
# 体力消耗/回复公式
# ============================================================================

def calc_vital_cost(
    train_idx: int,
    train_level: int,
) -> int:
    """计算训练的体力消耗

    从TRAINING_BASIC_VALUE读取，智力训练回复体力而非消耗。

    Args:
        train_idx: 训练类型(0-4)
        train_level: 训练等级(0-4)

    Returns:
        体力变化量（负数=消耗，正数=回复）
    """
    return TRAINING_BASIC_VALUE[train_idx][train_level][6]


def calc_rest_vital(
    motivation: int,
    is_great_success: bool = False,
) -> int:
    """计算お休み的体力回复量

    基础回复由やる気决定，大成功时额外+10

    Args:
        motivation: やる気(1-5)
        is_great_success: 是否大成功

    Returns:
        体力回复量
    """
    base = MOTIVATION_REST_VITAL.get(motivation, 60)
    if is_great_success:
        base += 10
    return base


def calc_great_success_prob(vital: int, max_vital: int) -> float:
    """お休み大成功概率

    体力越低，大成功概率越高

    Args:
        vital: 当前体力
        max_vital: 体力上限

    Returns:
        大成功概率(0.0-1.0)
    """
    ratio = vital / max_vital if max_vital > 0 else 1.0
    # 体力低时概率高（约30-40%），体力高时概率低（约10%）
    return max(0.1, 0.4 - 0.3 * ratio)


# ============================================================================
# 训练值计算
# ============================================================================

def calc_training_value(
    train_idx: int,
    train_level: int,
    motivation: int,
    vital: int,
    max_vital: int,
    bc_manager: BadConditionManager,
    support_bonus: Optional[List[int]] = None,
    friend_bonus: float = 1.0,
    scenario_bonus: Optional[List[int]] = None,
) -> Tuple[List[int], int]:
    """计算训练获得的属性值

    计算流程：
      1. 基础值 = TRAINING_BASIC_VALUE[train][level]
      2. やる気倍率
      3. 太り気味检查（Speed训练无效）
      4. 支援卡加成
      5. 友人卡加成
      6. 剧本加成
      7. 钳制到上限

    Args:
        train_idx: 训练类型(0-4)
        train_level: 训练等级(0-4)
        motivation: やる気(1-5)
        vital: 当前体力
        max_vital: 体力上限
        bc_manager: バッドコンディション管理器
        support_bonus: 支援卡加成[5维+pt]（6元素）
        friend_bonus: 友人卡倍率
        scenario_bonus: 剧本额外加成[5维+pt]（6元素）

    Returns:
        (训练值[速,耐,力,根,智,pt], 体力消耗)
    """
    # 基础值
    basic = TRAINING_BASIC_VALUE[train_idx][train_level]
    # [速,耐,力,根,智,pt,体力消耗]
    values = [basic[0], basic[1], basic[2], basic[3], basic[4], basic[5]]
    vital_cost = basic[6]

    # やる気倍率
    mot_mult = MOTIVATION_MULTIPLIER.get(motivation, 1.0)
    for i in range(6):
        values[i] = int(values[i] * mot_mult)

    # 太り気味：Speed训练无效
    if train_idx == 0 and bc_manager.is_speed_training_disabled():
        values[0] = 0  # Speed归零
        # pt保留

    # 支援卡加成
    if support_bonus:
        for i in range(min(6, len(support_bonus))):
            values[i] += support_bonus[i]

    # 友人卡倍率
    if friend_bonus != 1.0:
        for i in range(5):  # 属性部分
            values[i] = int(values[i] * friend_bonus)

    # 剧本加成
    if scenario_bonus:
        for i in range(min(6, len(scenario_bonus))):
            values[i] += scenario_bonus[i]

    # 确保非负
    values = [max(0, v) for v in values]

    return values, vital_cost


# ============================================================================
# お出かけ/外出效果
# ============================================================================

# 友人外出效果模板（通用）
FRIEND_OUTING_EFFECT = {
    'vital': 30,         # 体力回复
    'motivation': 1,     # やる気+1
    'friendship': 25,    # 羁绊+25
}

# 拉面杯お出かけ(304)额外效果
RAMEN_OUTING_EFFECT = {
    'kakushimi': 2,      # 隠し味の秘訣+2
}


# ============================================================================
# 公式层入口
# ============================================================================

class FormulaLayer:
    """公式层入口类

    聚合所有硬逻辑公式，供Game类调用。
    纯数据驱动一定出错，公式层提供确定性保证。
    """

    def __init__(self, bc_manager: BadConditionManager):
        self.bc = bc_manager

    # 失败率
    def fail_rate(self, vital, motivation, train_idx, train_level,
                  bias=0, is_race=False) -> int:
        return calc_fail_rate(
            vital, motivation, train_idx, train_level,
            self.bc, bias, is_race
        )

    # 体力消耗
    def vital_cost(self, train_idx, train_level) -> int:
        return calc_vital_cost(train_idx, train_level)

    # お休み回复
    def rest_vital(self, motivation, great_success=False) -> int:
        return calc_rest_vital(motivation, great_success)

    # 大成功概率
    def great_success_prob(self, vital, max_vital) -> float:
        return calc_great_success_prob(vital, max_vital)

    # 训练值
    def training_value(self, train_idx, train_level, motivation,
                       vital, max_vital, support_bonus=None,
                       friend_bonus=1.0, scenario_bonus=None):
        return calc_training_value(
            train_idx, train_level, motivation, vital, max_vital,
            self.bc, support_bonus, friend_bonus, scenario_bonus
        )

    # やる気倍率
    @staticmethod
    def motivation_multiplier(motivation: int) -> float:
        return MOTIVATION_MULTIPLIER.get(motivation, 1.0)

    # バッドコンディション回合处理
    def bc_on_turn_start(self, turn, rng):
        return self.bc.on_turn_start(turn, rng)

    # 保健室治愈
    def bc_heal_clinic(self):
        return self.bc.heal_by_clinic()

    # お休み治愈
    def bc_heal_rest(self, rng):
        return self.bc.heal_by_rest(rng)

    # 记者治愈
    def bc_heal_reporter(self):
        return self.bc.heal_lazy_by_reporter()
