"""
赛马娘AI训练框架 - バッドコンディション（负面状态）系统

建模育成过程中的6种负面状态及其获取/治愈机制。
基于2.5周年平衡补丁(2025.11.11)的改动完整建模。

状态列表：
  練習ベタ(Bad)   - 训练失败率+2%
  なまけ癖(Lazy)  - 概率跳过训练（2.5周年后：随机触发+冷却）
  太り気味(Fat)   - Speed训练无效
  片頭痛(Headache) - やる気不上升
  肌荒れ(Skin)    - 每回合概率やる気-1
  夜ふかし(LateBed)- 每回合体力-10

治愈方式：
  - 保健室：必消1个（2.5周年后确定性，非随机）
  - お休み：概率治愈夜ふかし/肌荒れ
  - 乙名史记者同行训练：治愈なまけ癖
"""

import random
from enum import IntEnum
from dataclasses import dataclass, field
from typing import List, Optional, Set


class BadConditionType(IntEnum):
    """バッドコンディション类型枚举"""
    BAD = 1          # 練習ベタ（练习下手）- 失败率+2%
    LAZY = 2         # なまけ癖（懒惰癖）- 概率跳过训练
    FAT = 3          # 太り気味（发胖）- Speed训练无效
    HEADACHE = 4     # 片頭痛（偏头痛）- やる気不上升
    SKIN = 5         # 肌荒れ（皮肤粗糙）- 每回合概率やる気-1
    LATE_BED = 6     # 夜ふかし（熬夜）- 每回合体力-10


# 类型→名称映射
BAD_CONDITION_NAMES = {
    BadConditionType.BAD: "練習ベタ",
    BadConditionType.LAZY: "なまけ癖",
    BadConditionType.FAT: "太り気味",
    BadConditionType.HEADACHE: "片頭痛",
    BadConditionType.SKIN: "肌荒れ",
    BadConditionType.LATE_BED: "夜ふかし",
}


@dataclass
class BadCondition:
    """单个バッドコンディション实例"""
    bc_type: BadConditionType    # 状态类型
    turn_acquired: int           # 获取回合
    cooldown_turns: int = 0      # 冷却回合（なまけ癖专用）

    @property
    def name(self) -> str:
        return BAD_CONDITION_NAMES.get(self.bc_type, "未知")


class BadConditionManager:
    """バッドコンディション管理器
    
    管理当前所有负面状态的获取、检查、治愈逻辑。
    2.5周年补丁改动：
      - 保健室必消1个（确定性）
      - なまけ癖随机触发+冷却
      - お休み概率治夜ふかし/肌荒れ
      - やる気下降事件不重复（5回合内）
    """

    def __init__(self):
        self.conditions: List[BadCondition] = []
        self._last_motivation_down_turn: int = -10  # 上次やる気下降回合

    # ===== 查询 =====

    def has(self, bc_type: BadConditionType) -> bool:
        """是否有指定类型的负面状态"""
        return any(c.bc_type == bc_type for c in self.conditions)

    def get(self, bc_type: BadConditionType) -> Optional[BadCondition]:
        """获取指定类型的负面状态实例"""
        for c in self.conditions:
            if c.bc_type == bc_type:
                return c
        return None

    @property
    def count(self) -> int:
        """当前负面状态数量"""
        return len(self.conditions)

    @property
    def types(self) -> Set[BadConditionType]:
        """当前所有负面状态类型"""
        return {c.bc_type for c in self.conditions}

    # ===== 效果查询 =====

    def get_fail_rate_bonus(self) -> int:
        """バッドコンディション导致的失败率加成（百分比）
        
        練習ベタ: +2%
        """
        if self.has(BadConditionType.BAD):
            return 2
        return 0

    def is_speed_training_disabled(self) -> bool:
        """Speed训练是否无效（太り気味）"""
        return self.has(BadConditionType.FAT)

    def is_motivation_blocked(self) -> bool:
        """やる気是否无法上升（片頭痛）"""
        return self.has(BadConditionType.HEADACHE)

    def should_skip_training(self, current_turn: int, rng: random.Random) -> bool:
        """なまけ癖是否导致跳过训练
        
        2.5周年后：随机触发+冷却机制
        冷却期间不触发，触发后有冷却期
        """
        bc = self.get(BadConditionType.LAZY)
        if bc is None:
            return False

        # 冷却中不触发
        if bc.cooldown_turns > 0:
            bc.cooldown_turns -= 1
            return False

        # 随机触发概率约40%（实测估算）
        if rng.random() < 0.4:
            bc.cooldown_turns = 3  # 触发后冷却3回合
            return True
        return False

    def get_skin_motivation_drain(self, rng: random.Random) -> int:
        """肌荒れ导致的やる気下降量
        
        每回合约35%概率やる気-1
        """
        if self.has(BadConditionType.SKIN) and rng.random() < 0.35:
            return -1
        return 0

    def get_late_bed_vital_drain(self) -> int:
        """夜ふかし导致的体力消耗"""
        if self.has(BadConditionType.LATE_BED):
            return -10
        return 0

    # ===== 获取 =====

    def acquire(self, bc_type: BadConditionType, current_turn: int) -> bool:
        """获取バッドコンディション
        
        同类型不重复获取（除了特殊标记）
        Returns: 是否新获取
        """
        if self.has(bc_type):
            return False  # 已有同类型，不重复

        cooldown = 0
        # なまけ癖初始冷却2回合
        if bc_type == BadConditionType.LAZY:
            cooldown = 2

        self.conditions.append(BadCondition(
            bc_type=bc_type,
            turn_acquired=current_turn,
            cooldown_turns=cooldown,
        ))
        return True

    def try_acquire_random(self, current_turn: int, rng: random.Random) -> Optional[BadConditionType]:
        """随机事件可能获取バッドコンディション
        
        基于实测统计的获取概率（每回合）：
          - 練習ベタ: ~3%
          - なまけ癖: ~2%
          - 太り気味: ~1.5%（食物事件选下）
          - 片頭痛: ~2%
          - 肌荒れ: ~1%（连续出走后更高）
          - 夜ふかし: ~3%（お休み后概率）
        """
        # 按优先级尝试获取
        acquire_table = [
            (BadConditionType.BAD, 0.03),
            (BadConditionType.LAZY, 0.02),
            (BadConditionType.FAT, 0.015),
            (BadConditionType.HEADACHE, 0.02),
            (BadConditionType.SKIN, 0.01),
            (BadConditionType.LATE_BED, 0.03),
        ]
        for bc_type, prob in acquire_table:
            if not self.has(bc_type) and rng.random() < prob:
                self.acquire(bc_type, current_turn)
                return bc_type
        return None

    # ===== 治愈 =====

    def heal_by_clinic(self) -> Optional[BadConditionType]:
        """保健室治愈
        
        2.5周年后：必消1个（确定性，优先级：片頭痛 > 練習ベタ > なまけ癖 > 其他）
        Returns: 被治愈的类型
        """
        if not self.conditions:
            return None

        # 治愈优先级
        priority = [
            BadConditionType.HEADACHE,
            BadConditionType.BAD,
            BadConditionType.LAZY,
            BadConditionType.LATE_BED,
            BadConditionType.SKIN,
            BadConditionType.FAT,
        ]
        for bc_type in priority:
            for i, c in enumerate(self.conditions):
                if c.bc_type == bc_type:
                    self.conditions.pop(i)
                    return bc_type
        return None

    def heal_by_rest(self, rng: random.Random) -> List[BadConditionType]:
        """お休み治愈
        
        概率治愈：
          - 夜ふかし: 50%
          - 肌荒れ: 40%
        Returns: 被治愈的类型列表
        """
        healed = []
        # 夜ふかし
        if self.has(BadConditionType.LATE_BED) and rng.random() < 0.5:
            self._remove(BadConditionType.LATE_BED)
            healed.append(BadConditionType.LATE_BED)
        # 肌荒れ
        if self.has(BadConditionType.SKIN) and rng.random() < 0.4:
            self._remove(BadConditionType.SKIN)
            healed.append(BadConditionType.SKIN)
        return healed

    def heal_lazy_by_reporter(self) -> bool:
        """记者同行训练治愈なまけ癖"""
        if self.has(BadConditionType.LAZY):
            self._remove(BadConditionType.LAZY)
            return True
        return False

    def _remove(self, bc_type: BadConditionType) -> bool:
        """移除指定类型"""
        for i, c in enumerate(self.conditions):
            if c.bc_type == bc_type:
                self.conditions.pop(i)
                return True
        return False

    # ===== 回合处理 =====

    def on_turn_start(self, current_turn: int, rng: random.Random) -> dict:
        """回合开始时处理バッドコンディション效果
        
        Returns: 效果字典 {
            'vital_drain': int,       # 体力消耗（夜ふかし）
            'motivation_drain': int,  # やる気下降（肌荒れ）
            'skip_training': bool,    # 是否跳过训练（なまけ癖）
        }
        """
        result = {
            'vital_drain': 0,
            'motivation_drain': 0,
            'skip_training': False,
        }

        # 夜ふかし：体力-10
        result['vital_drain'] = self.get_late_bed_vital_drain()

        # 肌荒れ：概率やる気-1
        result['motivation_drain'] = self.get_skin_motivation_drain(rng)

        # なまけ癖：可能跳过训练
        result['skip_training'] = self.should_skip_training(current_turn, rng)

        return result

    def can_motivation_decrease(self, current_turn: int) -> bool:
        """やる気下降事件是否可以触发
        
        2.5周年后：5回合内不重复
        """
        return current_turn - self._last_motivation_down_turn >= 5

    def record_motivation_decrease(self, current_turn: int):
        """记录やる気下降事件触发"""
        self._last_motivation_down_turn = current_turn

    # ===== 序列化 =====

    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            'conditions': [
                {'type': int(c.bc_type), 'name': c.name,
                 'turn': c.turn_acquired, 'cooldown': c.cooldown_turns}
                for c in self.conditions
            ],
            'last_motivation_down_turn': self._last_motivation_down_turn,
        }

    def __repr__(self) -> str:
        if not self.conditions:
            return "バッドコンディション: なし"
        names = [c.name for c in self.conditions]
        return f"バッドコンディション: {', '.join(names)}"
