"""
赛马娘AI训练框架 - Ramen剧本（拉面杯）

scenario_id=14，2026年6月新实装的剧本。
包含試食会、地域、隠し味の秘訣、コツ等特有机制。

关键机制：
  - 試食会(EffectCategory=1): 类似比赛的奖励事件
  - 地域(EffectCategory=2): 地域贡献度，影响隠し味获取
  - 隠し味の秘訣(EffectCategory=4): 核心道具，お出かけ(304)获取+2
  - コツ: 训练后的技能hint触发
  - command_id 601-605: 五训（非通用101-106）
  - command_id 304: お出かけ（友人外出）
  - FeelingType: 拉面杯专属feeling系统
  - CheckPoint: 目标达成奖励

参考：
  - SO实测数据结构 WorkSingleModeScenarioRamenDataSet
  - EffectCategory实测映射: 1=試食会 2=地域 4=隠し味
"""

import random
from typing import List, Optional, Dict
from simulator.scenarios.base import ScenarioBase


class RamenScenario(ScenarioBase):
    """Ramen剧本（拉面杯）

    特有机制：
    - 隠し味の秘訣：核心资源，通过お出かけ/試食会/地域获取
    - 試食会：周期性奖励事件（类似比赛但给隠し味）
    - 地域贡献：影响试食会奖励和隠し味倍率
    - コツ系统：训练后概率触发，提供技能hint
    - CheckPoint：目标达成给属性+隠し味奖励
    - Feeling：拉面杯专属feeling系统（不同于やる気）
    """

    SCENARIO_ID = 14
    SCENARIO_NAME = "Ramen"

    def __init__(self):
        # 隠し味の秘訣数量
        self.kakushimi_count: int = 0
        # 地域贡献度（按区域ID）
        self.region_points: Dict[int, int] = {}
        # 試食会次数
        self.tasting_count: int = 0
        # コツ等级（按训练类型）
        self.kotsu_level: List[int] = [0] * 5
        # CheckPoint进度
        self.checkpoint_pt: int = 0
        self.expected_checkpoint_pt: int = 0
        # Feeling状态
        self.feeling_values: Dict[int, int] = {}
        self.special_feeling_num: int = 0

    def on_turn_start(self, game, rng: random.Random):
        """回合开始时Ramen特定逻辑"""
        # コツ系统：每回合概率提升（基于训练次数）
        for i in range(5):
            if game.train_level_count[i] > 0 and rng.random() < TASTING_EVENT_PROB:
                self.kotsu_level[i] = min(3, self.kotsu_level[i] + 1)

    def on_turn_end(self, game, rng: random.Random):
        """回合结束时的Ramen特定逻辑"""
        # CheckPoint检查
        if self.checkpoint_pt >= self.expected_checkpoint_pt and self.expected_checkpoint_pt > 0:
            self._grant_checkpoint_reward(game, rng)
            self.expected_checkpoint_pt = self._next_checkpoint_target()

    def modify_training_value(self, game, train_idx: int, train_value: List[int]) -> List[int]:
        """Ramen剧本训练值修改

        加成来源：
        - 隠し味の秘訣倍率（隠し味越多，训练值越高）
        - コツ等级加成
        - Feeling加成
        """
        # 隠し味倍率：每10个隠し味 +1% 训练值（上限30%）
        kakushimi_bonus = min(KAKUSHIMI_MAX_BONUS, self.kakushimi_count * KAKUSHIMI_PER_BONUS)
        if kakushimi_bonus > 0:
            for i in range(5):
                train_value[i] = int(train_value[i] * (1.0 + kakushimi_bonus))

        # コツ加成：每级+2对应属性
        if train_idx < 5 and self.kotsu_level[train_idx] > 0:
            train_value[train_idx] += self.kotsu_level[train_idx] * 2

        return train_value

    def get_extra_actions(self, game) -> List[int]:
        """Ramen特有的额外动作"""
        # お出かけ(304) — 在action中已映射为OUTGOING
        return []

    def calculate_score(self, game) -> int:
        """Ramen剧本评分计算

        隠し味の秘訣贡献评分
        """
        # 隠し味每个值约80-120分（实测估算）
        return self.kakushimi_count * 100

    # ===== 隠し味获取 =====

    def add_kakushimi(self, count: int):
        """获取隠し味の秘訣"""
        self.kakushimi_count += count

    def get_kakushimi_from_outing(self) -> int:
        """お出かけ(304)获取隠し味

        每次外出固定+2
        """
        return 2

    def get_kakushimi_from_tasting(self, rng: random.Random) -> int:
        """試食会获取隠し味

        基础1-3个，地域贡献度高时更多
        """
        base = rng.randint(1, 3)
        # 地域贡献加成
        region_bonus = sum(1 for v in self.region_points.values() if v >= 50)
        return base + region_bonus

    # ===== 試食会 =====

    def maybe_trigger_tasting(self, game, rng: random.Random) -> bool:
        """检查是否触发試食会

        試食会在特定回合触发（类似比赛日）
        """
        # 試食会触发回合（实测：约每12回合一次）
        tasting_turns = [12, 24, 36, 48, 60]
        if game.turn in tasting_turns:
            self.tasting_count += 1
            kakushimi = self.get_kakushimi_from_tasting(rng)
            self.add_kakushimi(kakushimi)
            # 試食会给属性加成
            game.add_all_status(5 + self.tasting_count)
            game.skill_pt += 15 + self.tasting_count * 5
            return True
        return False

    # ===== 地域贡献 =====

    def add_region_point(self, region_id: int, value: int):
        """增加地域贡献度"""
        self.region_points[region_id] = self.region_points.get(region_id, 0) + value

    # ===== CheckPoint =====

    def _grant_checkpoint_reward(self, game, rng: random.Random):
        """达成CheckPoint给奖励"""
        # 属性加成（递增）
        bonus = 5 + self.checkpoint_pt // 100
        game.add_all_status(bonus)
        game.skill_pt += bonus * 3
        # 隠し味奖励
        self.add_kakushimi(1)

    def _next_checkpoint_target(self) -> int:
        """下一个CheckPoint目标"""
        # 目标递增（100, 250, 450, 700, 1000...）
        targets = [100, 250, 450, 700, 1000, 1350, 1750, 2200]
        idx = 0
        for i, t in enumerate(targets):
            if self.checkpoint_pt < t:
                idx = i
                break
        return targets[min(idx, len(targets) - 1)]

    # ===== Feeling系统 =====

    def add_feeling(self, feeling_type: int, value: int):
        """添加Feeling值"""
        self.feeling_values[feeling_type] = self.feeling_values.get(feeling_type, 0) + value

    # ===== 序列化 =====

    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            'kakushimi_count': self.kakushimi_count,
            'region_points': dict(self.region_points),
            'tasting_count': self.tasting_count,
            'kotsu_level': list(self.kotsu_level),
            'checkpoint_pt': self.checkpoint_pt,
            'expected_checkpoint_pt': self.expected_checkpoint_pt,
            'feeling_values': dict(self.feeling_values),
            'special_feeling_num': self.special_feeling_num,
        }

    def __repr__(self) -> str:
        return (f"Ramen: 隠し味={self.kakushimi_count} 試食会={self.tasting_count} "
                f"コツ={self.kotsu_level} CP={self.checkpoint_pt}")
