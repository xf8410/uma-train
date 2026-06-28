"""
赛马娘AI训练框架 - Dreams剧本（育马者杯/Beyond Dreams）

scenario_id=13，目前优先实现的剧本。
包含机甲升级、UGE比赛、overdrive等特有机制。
"""

import random
from typing import List, Optional
from simulator.scenarios.base import ScenarioBase


class DreamsScenario(ScenarioBase):
    """Dreams剧本（凯旋门/机甲剧本）
    
    特有机制：
    - 研究等级（rivalLv）- 影响训练加成
    - EN点数 - 用于升级机甲（头/胸/脚）
    - UGE比赛 - 定期检查研究等级，给予奖励
    - Overdrive - 消耗能量开启训练强化
    - 齿轮(Gear) - 额外训练加成
    """
    
    SCENARIO_ID = 13
    SCENARIO_NAME = "Beyond Dreams"
    
    def on_turn_start(self, game, rng: random.Random):
        """回合开始时的Dreams特定逻辑"""
        # 检查是否需要开启overdrive（URA期间自动开启）
        if game.turn >= 72 and not game.mecha_any_lose:
            game.mecha_overdrive_energy = 3
    
    def on_turn_end(self, game, rng: random.Random):
        """回合结束时的Dreams特定逻辑"""
        pass
    
    def modify_training_value(self, game, train_idx: int, train_value: List[int]) -> List[int]:
        """Dreams剧本训练值修改（已在Game类中实现）"""
        return train_value
    
    def get_extra_actions(self, game) -> List[int]:
        """获取Dreams特有的额外动作（overdrive、升级等）"""
        return []
    
    def calculate_score(self, game) -> int:
        """Dreams剧本评分计算"""
        # 使用Game类中已实现的评分
        return game.final_score()
    
    @staticmethod
    def get_uge_turns() -> List[int]:
        """获取UGE比赛回合"""
        return [1, 23, 35, 47, 59, 71]
    
    @staticmethod
    def get_mecha_upgrade_bonus(upgrade_level: int) -> float:
        """机甲升级对应的训练加成"""
        return {0: 0, 1: 10, 2: 18, 3: 26, 4: 33, 5: 40}.get(upgrade_level, 0)
