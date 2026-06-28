"""
赛马娘AI训练框架 - 手写评分逻辑

参考 umaai-rs 的 handwritten_evaluator.rs，实现基于规则的启发式评估器。
用于MCTS搜索时没有神经网络时的fallback。
"""

import math
import random
from typing import List, Optional
from simulator.game import Game
from simulator.action import Action, TrainActionType, GameStage
from simulator.person import PersonType
from config import TOTAL_TURN


# ============================================================================
# 常量定义（参考 umaai-rs handwritten_evaluator.rs 和 C++ HandwrittenLogic）
# ============================================================================

# 属性权重 [速度, 耐力, 力量, 根性, 智力]
STATUS_WEIGHTS = [7.0, 8.0, 8.0, 8.0, 6.0]

# 没带卡的属性权重
ABSENT_WEIGHT = 2.0

# 训练类型偏好 [速, 耐, 力, 根, 智]
TRAIN_TYPE_BONUS = [20.0, 10.0, 30.0, 30.0, 20.0]

# 智力训练人头阈值
ABSENT_HEAD_THRESHOLD = 2

# 前期回合阈值
EARLY_TURN_THRESHOLD = 12

# 控属性预留空间因子
RESERVE_STATUS_FACTOR = 40.0

# 羁绊基础价值
JIBAN_VALUE = 12.0

# 体力价值因子
VITAL_FACTOR_START = 3.5
VITAL_FACTOR_END = 2.0

# 失败惩罚
SMALL_FAIL_VALUE = -1500.0
BIG_FAIL_VALUE = -1800.0

# 外出加成
OUTGOING_BONUS_IF_NOT_FULL_MOTIVATION = 200.0

# 休息加成
REST_BASE_VALUE = -100.0

# 比赛加成
RACE_BASE_BONUS = 450.0
NON_TARGET_RACE_BONUS = 180.0


# ============================================================================
# 辅助函数
# ============================================================================

def status_soft_function(x: float, reserve: float) -> float:
    """属性上限软约束函数
    
    当属性接近上限时，进一步增加属性的边际价值递减。
    """
    if reserve <= 0.0:
        return min(x, 0.0)
    
    reserve_inv_x2 = 1.0 / (2.0 * reserve)
    
    if x >= 0.0:
        return 0.0
    elif x > -reserve:
        return -x * x * reserve_inv_x2 * 0.5
    else:
        return x + 0.25 * reserve


def vital_evaluation(vital: int, max_vital: int) -> float:
    """体力分段评估函数
    
    体力越低，边际价值越高（更想回复体力）。
    """
    if vital <= 50:
        return 2.0 * vital
    elif vital <= 70:
        return 1.5 * (vital - 50) + vital_evaluation(50, max_vital)
    elif vital <= max_vital:
        return 1.0 * (vital - 70) + vital_evaluation(70, max_vital)
    else:
        return vital_evaluation(max_vital, max_vital)


def calc_vital_factor(turn: int, max_turn: int = TOTAL_TURN) -> float:
    """计算当前回合的体力价值因子
    
    前期体力价值高，后期体力价值低。
    """
    return VITAL_FACTOR_START + (turn / max_turn) * (VITAL_FACTOR_END - VITAL_FACTOR_START)


class HandwrittenEvaluator:
    """
    手写启发式评估器
    
    参考 umaai-rs 的 HandwrittenEvaluator，实现基于规则的评估。
    用于MCTS搜索时没有神经网络的fallback场景。
    """
    
    def __init__(
        self,
        weights: Optional[List[float]] = None,
        skill_weight: float = 0.5,
        vital_threshold: int = 55,
        shining_bonus: float = 35.0,
    ):
        """
        Args:
            weights: 属性权重 [速,耐,力,根,智]
            skill_weight: 技能点权重
            vital_threshold: 体力阈值
            shining_bonus: 彩圈加成
        """
        self.weights = weights or list(STATUS_WEIGHTS)
        self.skill_weight = skill_weight
        self.vital_threshold = vital_threshold
        self.shining_bonus = shining_bonus
    
    def select_action(self, game: Game, rng: random.Random) -> Action:
        """根据手写逻辑选择动作
        
        Args:
            game: 游戏状态
            rng: 随机数生成器
            
        Returns:
            选择的动作
        """
        best_action = None
        best_value = float('-inf')
        
        # 遍历所有可能的动作
        for action_int in range(Action.MAX_ACTION_TYPE):
            action = Action.from_int(action_int)
            if not game.is_legal(action):
                continue
            
            value = self._evaluate_action(game, action, rng)
            if value > best_value:
                best_value = value
                best_action = action
        
        if best_action is None:
            # 没有合法动作，选择休息
            return Action(type=GameStage.BEFORE_TRAIN, train=TrainActionType.REST)
        
        return best_action
    
    def evaluate(self, game: Game) -> int:
        """评估游戏状态的最终得分
        
        Args:
            game: 游戏状态
            
        Returns:
            估算的最终得分
        """
        return game.final_score()
    
    def _evaluate_action(self, game: Game, action: Action, rng: random.Random) -> float:
        """评估单个动作的价值"""
        if action.type == GameStage.BEFORE_MECHA_UPGRADE:
            return self._evaluate_upgrade(game, action)
        
        train = action.train
        if train == TrainActionType.REST:
            return self._evaluate_rest(game)
        elif train == TrainActionType.OUTGOING:
            return self._evaluate_outgoing(game)
        elif train == TrainActionType.RACE:
            return self._evaluate_race(game)
        elif 0 <= train <= 4:
            return self._evaluate_training(game, train)
        
        return -1000.0
    
    def _evaluate_training(self, game: Game, train: int) -> float:
        """评估训练动作"""
        score = 0.0
        
        # 属性收益
        for i in range(5):
            limit = game.five_status_limit[i]
            remain = limit - game.five_status[i] - 45  # 预留URA加成
            gain = game.train_value[train][i]
            
            remain_turn = max(0, TOTAL_TURN - game.turn - 1)
            total_turn = TOTAL_TURN
            reserve = RESERVE_STATUS_FACTOR * remain_turn * (1.0 - remain_turn / (total_turn * 2.0))
            
            s0 = status_soft_function(-remain, reserve)
            s1 = status_soft_function(gain - remain, reserve)
            
            weight = self.weights[i] if game.persons[train].card_param.card_type == train else ABSENT_WEIGHT
            score += weight * (s1 - s0)
        
        # pt收益
        score += game.train_value[train][5] * self.skill_weight
        
        # 体力价值评估
        vital_factor = calc_vital_factor(game.turn)
        vital_before = vital_evaluation(game.vital, game.max_vital)
        vital_after = max(0, min(game.vital + game.train_vital_change[train], game.max_vital))
        vital_after_value = vital_evaluation(vital_after, game.max_vital)
        score += vital_factor * (vital_after_value - vital_before)
        
        # 失败率惩罚
        fail_rate = game.fail_rate[train]
        big_fail_prob = fail_rate if fail_rate >= 20 else 0
        fail_value_avg = 0.01 * big_fail_prob * BIG_FAIL_VALUE + (1 - 0.01 * big_fail_prob) * SMALL_FAIL_VALUE
        score = 0.01 * fail_rate * fail_value_avg + (1 - 0.01 * fail_rate) * score
        
        # 彩圈加成
        shining_count = sum(1 for h in range(5) if game.person_distribution[train][h] >= 0 and game.is_card_shining(game.person_distribution[train][h], train))
        if shining_count > 0:
            score *= 1.0 + shining_count * 0.15
            score += shining_count * self.shining_bonus
        
        # 羁绊价值
        for h in range(5):
            pid = game.person_distribution[train][h]
            if pid < 0:
                break
            if pid >= 6:
                continue
            p = game.persons[pid]
            if p.person_type == PersonType.CARD:
                if p.friendship < 80:
                    jiban_add = min(7.0, 80 - p.friendship)
                    score += jiban_add * JIBAN_VALUE
                if p.is_hint:
                    hint_bonus = sum(STATUS_WEIGHTS) * 1.6
                    score += hint_bonus
        
        # 训练类型偏好
        score += TRAIN_TYPE_BONUS[train]
        
        # 没带卡的训练惩罚
        # 检查是否有该类型的卡
        has_card = any(
            game.persons[i].card_param.card_type == train
            for i in range(6)
            if game.persons[i].person_type == PersonType.CARD
        )
        if not has_card:
            score -= 100.0
            head_count = sum(1 for h in range(5) if game.person_distribution[train][h] >= 0)
            if game.turn < EARLY_TURN_THRESHOLD and train == 4 and head_count >= 3:
                score += 120.0
            if head_count >= ABSENT_HEAD_THRESHOLD + 1:
                score += 80.0
        
        return score
    
    def _evaluate_rest(self, game: Game) -> float:
        """评估休息"""
        vital_factor = calc_vital_factor(game.turn)
        vital_before = vital_evaluation(game.vital, game.max_vital)
        vital_after = min(game.vital + 50, game.max_vital)
        vital_after_value = vital_evaluation(vital_after, game.max_vital)
        return vital_factor * (vital_after_value - vital_before)
    
    def _evaluate_outgoing(self, game: Game) -> float:
        """评估外出"""
        score = 0.0
        vital_factor = calc_vital_factor(game.turn)
        vital_before = vital_evaluation(game.vital, game.max_vital)
        
        if game.friend_type != 0 and game.friend_stage >= 2 and game.friend_outgoing_used < 5:
            # 友人外出
            score += 400.0
            vital_after = min(game.vital + 30, game.max_vital)
            vital_after_value = vital_evaluation(vital_after, game.max_vital)
            score += vital_factor * (vital_after_value - vital_before)
        else:
            # 普通外出
            vital_after = min(game.vital + 10, game.max_vital)
            vital_after_value = vital_evaluation(vital_after, game.max_vital)
            score += vital_factor * (vital_after_value - vital_before)
        
        if game.motivation < 5:
            score += OUTGOING_BONUS_IF_NOT_FULL_MOTIVATION
        
        return score
    
    def _evaluate_race(self, game: Game) -> float:
        """评估比赛"""
        if game.is_racing:
            return RACE_BASE_BONUS
        elif game.is_race_available():
            return NON_TARGET_RACE_BONUS
        return -1000.0
    
    def _evaluate_upgrade(self, game: Game, action: Action) -> float:
        """评估升级动作"""
        # 简化：优先头和胸
        return action.mecha_head * 10 + action.mecha_chest * 10
