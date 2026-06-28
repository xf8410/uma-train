"""
赛马娘AI训练框架 - 剧本基类

定义剧本的通用接口，具体剧本继承此基类实现特定逻辑。
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import random


class ScenarioBase(ABC):
    """剧本基类"""
    
    # 剧本ID
    SCENARIO_ID: int = 0
    SCENARIO_NAME: str = "基础"
    
    @abstractmethod
    def on_turn_start(self, game, rng: random.Random):
        """回合开始时的剧本特定逻辑"""
        pass
    
    @abstractmethod
    def on_turn_end(self, game, rng: random.Random):
        """回合结束时的剧本特定逻辑"""
        pass
    
    @abstractmethod
    def modify_training_value(self, game, train_idx: int, train_value: List[int]) -> List[int]:
        """修改训练值（剧本加成）"""
        return train_value
    
    @abstractmethod
    def get_extra_actions(self, game) -> List[int]:
        """获取剧本特有的额外动作"""
        return []
    
    @abstractmethod
    def calculate_score(self, game) -> int:
        """剧本特定的评分计算"""
        return 0
