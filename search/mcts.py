"""
赛马娘AI训练框架 - MCTS搜索核心

参考 UmaAi 的 Search.h/cpp，实现蒙特卡洛树搜索。
支持多阶段搜索（先粗后精）、batch推理、手写逻辑fallback。
修复P1-11：策略软化使用正确的温度softmax替代错误的exp(x/N/delta)。
"""

import math
import random
from copy import deepcopy
from typing import List, Optional, Tuple

from .search_result import SearchResult, ModelOutputValue
from simulator.game import Game
from simulator.action import Action, TrainActionType, GameStage
from model.handwritten import HandwrittenEvaluator
from config import (
    TOTAL_TURN, EXPECTED_SEARCH_STDEV,
    SEARCH_STAGE_NUM, SEARCH_FACTOR_STAGE, SEARCH_THRESHOLD_STDEV_STAGE,
    NN_OUTPUT_C_POLICY, NN_OUTPUT_C_VALUE,
    MCTS_POLICY_TEMPERATURE,
)


def adjust_radical_factor(max_rf: float, turn: int) -> float:
    """根据回合数调整激进度
    
    回合越靠后，激进度越低（更保守）。
    """
    remain_turns = TOTAL_TURN - turn
    factor = (remain_turns / TOTAL_TURN) ** 0.5
    return factor * max_rf


class SearchParam:
    """搜索参数"""
    
    def __init__(
        self,
        search_single_max: int = 256,
        search_total_max: int = 0,
        search_group_size: int = 128,
        search_cpuct: float = 1.0,
        max_depth: int = 10,
        max_radical_factor: float = 5.0,
    ):
        self.search_single_max = search_single_max
        self.search_total_max = search_total_max
        self.search_group_size = search_group_size
        self.search_cpuct = search_cpuct
        self.max_depth = max_depth
        self.max_radical_factor = max_radical_factor


class MCTS:
    """
    蒙特卡洛树搜索
    
    参考 UmaAi 的 Search 类。对于每个可能的动作进行蒙特卡洛模拟，
    收集分数分布，最终选择最优动作。
    
    支持两种评估模式：
    1. 神经网络评估（需要提供模型）
    2. 手写逻辑评估（fallback，不依赖神经网络）
    """
    
    def __init__(
        self,
        model=None,           # 神经网络模型（可选）
        batch_size: int = 16,
        evaluator: Optional[HandwrittenEvaluator] = None,
    ):
        """
        Args:
            model: PyTorch模型，为None时使用手写逻辑
            batch_size: 批推理大小
            evaluator: 手写评估器，为None时创建默认的
        """
        self.model = model
        self.batch_size = batch_size
        self.evaluator = evaluator or HandwrittenEvaluator()
        self.all_action_results: List[SearchResult] = []
        self.root_game: Optional[Game] = None

    def run_search(
        self,
        game: Game,
        rng: random.Random,
        param: Optional[SearchParam] = None,
    ) -> Action:
        """对当前局面执行MCTS搜索，返回最优动作
        
        Args:
            game: 当前游戏状态
            rng: 随机数生成器
            param: 搜索参数
            
        Returns:
            最优动作
        """
        if param is None:
            param = SearchParam()

        self.root_game = deepcopy(game)
        radical_factor = adjust_radical_factor(param.max_radical_factor, self.root_game.turn)

        # 初始化搜索结果
        max_action_type = Action.MAX_ACTION_TYPE
        self.all_action_results = [SearchResult() for _ in range(max_action_type)]

        for action_int in range(max_action_type):
            action = Action.from_int(action_int)
            self.all_action_results[action_int].clear()
            self.all_action_results[action_int].is_legal = self.root_game.is_legal(action)

        # 第一轮：每个合法动作搜索一组
        for action_int in range(max_action_type):
            if not self.all_action_results[action_int].is_legal:
                continue
            action = Action.from_int(action_int)
            self._search_single_action(
                param.search_group_size, rng,
                self.all_action_results[action_int], action, param
            )

        total_search_n = sum(
            param.search_group_size for r in self.all_action_results if r.is_legal
        )

        # 后续轮次：分配计算量到searchValue最大的动作
        while param.search_group_size < param.search_single_max:
            best_search_value = -1e4
            best_action_int = -1

            for action_int in range(max_action_type):
                if not self.all_action_results[action_int].is_legal:
                    continue
                result = self.all_action_results[action_int]
                value = result.get_weighted_mean_score(radical_factor).value
                n = result.num
                if n <= 0:
                    continue
                
                tn = float(total_search_n)
                policy = 1.0  # 常数1，马娘搜索不需要神经网络policy
                search_value = value + param.search_cpuct * policy * EXPECTED_SEARCH_STDEV * math.sqrt(tn) / n
                
                if search_value > best_search_value:
                    best_search_value = search_value
                    best_action_int = action_int

            if best_action_int < 0:
                break

            action = Action.from_int(best_action_int)
            self._search_single_action(
                param.search_group_size, rng,
                self.all_action_results[best_action_int], action, param
            )
            total_search_n += param.search_group_size

            if self.all_action_results[best_action_int].num >= param.search_single_max:
                break
            if param.search_total_max > 0 and total_search_n >= param.search_total_max:
                break

        # 选择最优动作
        best_value = -1e4
        best_action_int = -1

        for action_int in range(max_action_type):
            if not self.all_action_results[action_int].is_legal:
                continue
            value = self.all_action_results[action_int].get_weighted_mean_score(radical_factor)
            if value.value > best_value:
                best_value = value.value
                best_action_int = action_int

        best_action = Action.from_int(best_action_int)
        return best_action

    def _search_single_action(
        self,
        search_n: int,
        rng: random.Random,
        search_result: SearchResult,
        action: Action,
        param: SearchParam,
    ):
        """对单个动作进行蒙特卡洛搜索"""
        for _ in range(search_n):
            # 复制游戏状态
            game_copy = deepcopy(self.root_game)
            
            # 应用动作
            game_copy.apply_action(rng, action)

            # 蒙特卡洛模拟到结束或最大深度
            for depth in range(param.max_depth):
                if game_copy.is_end():
                    break
                # 选择动作（用策略）
                next_action = self._select_action_by_policy(game_copy, rng)
                game_copy.apply_action(rng, next_action)

            # 评估最终状态
            if game_copy.is_end():
                score = game_copy.final_score()
                v = ModelOutputValue(
                    score_mean=float(score),
                    score_stdev=0.0,
                    value=float(score)
                )
            else:
                # 用评估器估算
                v = self._evaluate_game(game_copy, rng)

            search_result.add_result(v)

    def _select_action_by_policy(self, game: Game, rng: random.Random) -> Action:
        """根据策略选择动作
        
        如果有神经网络模型，使用网络输出策略；
        否则使用手写逻辑。
        """
        if self.model is not None:
            # 使用神经网络推理（简化版，实际应该batch推理）
            return self._select_action_by_nn(game, rng)
        else:
            # 使用手写逻辑
            return self.evaluator.select_action(game, rng)

    def _select_action_by_nn(self, game: Game, rng: random.Random) -> Action:
        """使用神经网络策略选择动作"""
        import torch
        from model.nn_input import encode_game_state

        # 编码游戏状态
        nn_input = encode_game_state(game)
        nn_input_tensor = torch.FloatTensor(nn_input).unsqueeze(0)

        # 推理
        with torch.no_grad():
            output = self.model(nn_input_tensor)
            policy_logits = output[0, :NN_OUTPUT_C_POLICY]

        # 掩除不合法动作
        for action_int in range(Action.MAX_ACTION_TYPE):
            action = Action.from_int(action_int)
            if not game.is_legal(action):
                policy_logits[action_int] = -1e7

        # softmax
        probs = torch.softmax(policy_logits, dim=0).numpy()

        # 按概率采样
        legal_indices = [i for i in range(len(probs)) if probs[i] > 1e-8]
        if not legal_indices:
            # fallback到手写逻辑
            return self.evaluator.select_action(game, rng)

        legal_probs = [probs[i] for i in legal_indices]
        total = sum(legal_probs)
        if total <= 0:
            chosen_idx = rng.choice(legal_indices)
        else:
            legal_probs = [p / total for p in legal_probs]
            chosen_idx = rng.choices(legal_indices, weights=legal_probs, k=1)[0]

        return Action.from_int(chosen_idx)

    def _evaluate_game(self, game: Game, rng: random.Random) -> ModelOutputValue:
        """评估游戏状态
        
        如果有神经网络模型，使用模型评估；
        否则使用手写逻辑估算。
        """
        if self.model is not None:
            import torch
            from model.nn_input import encode_game_state

            nn_input = encode_game_state(game)
            nn_input_tensor = torch.FloatTensor(nn_input).unsqueeze(0)

            with torch.no_grad():
                output = self.model(nn_input_tensor)
                value_output = output[0, NN_OUTPUT_C_POLICY:]

            # 反归一化
            score_mean = value_output[0].item() * 300 + 38000
            score_stdev = value_output[1].item() * 150
            optimistic = value_output[2].item() * 300 + 38000

            return ModelOutputValue(
                score_mean=score_mean,
                score_stdev=max(0, score_stdev),
                value=optimistic
            )
        else:
            # 手写逻辑评估
            score = self.evaluator.evaluate(game)
            progress = game.turn / TOTAL_TURN
            stdev = 500.0 * (1.0 - progress) + 100.0
            return ModelOutputValue(
                score_mean=float(score),
                score_stdev=stdev,
                value=float(score)
            )

    def print_search_result(self, param: SearchParam, show_search_num: bool = False):
        """打印搜索结果"""
        radical_factor = adjust_radical_factor(param.max_radical_factor, self.root_game.turn)
        for action_int in range(Action.MAX_ACTION_TYPE):
            action = Action.from_int(action_int)
            result = self.all_action_results[action_int]
            if not result.is_legal:
                continue
            value = result.get_weighted_mean_score(radical_factor)
            msg = f"{action}: {int(value.value)}"
            if show_search_num:
                msg += f", searchNum={result.num}"
            print(msg)

    def export_training_sample(self, temperature: float = None) -> dict:
        """导出训练样本
        
        修复P1-11：使用正确的温度softmax替代错误的exp(visit/total/delta)公式。
        原公式 exp(visit/N/delta) 当delta=100时所有动作exp值几乎相同，
        策略退化为均匀分布。
        
        正确做法：先归一化为访问比例（logits），再用温度参数做softmax。
        
        Args:
            temperature: 策略温度，None时使用配置默认值MCTS_POLICY_TEMPERATURE
            
        Returns:
            包含 nn_input, policy, value 的字典
        """
        from model.nn_input import encode_game_state

        if temperature is None:
            temperature = MCTS_POLICY_TEMPERATURE

        # 编码当前局面
        nn_input = encode_game_state(self.root_game)

        # 提取策略：统计每个动作的访问次数
        policy = [0.0] * NN_OUTPUT_C_POLICY
        total_visit = 0
        
        for action_int in range(Action.MAX_ACTION_TYPE):
            result = self.all_action_results[action_int]
            if not result.is_legal:
                continue
            if action_int < NN_OUTPUT_C_POLICY:
                policy[action_int] = result.num
                total_visit += result.num

        # 策略软化：正确的温度softmax
        if total_visit > 0:
            # 1. 归一化为访问比例（logits）
            logits = [p / total_visit for p in policy]
            
            # 2. 应用温度参数：logits / temperature
            #    temperature越大策略越均匀，越小越尖锐
            scaled_logits = [l / temperature for l in logits]
            
            # 3. 数值稳定的softmax：减去最大值防止溢出
            max_logit = max(scaled_logits)
            exp_vals = [math.exp(sl - max_logit) for sl in scaled_logits]
            total_exp = sum(exp_vals)
            
            if total_exp > 0:
                for i in range(NN_OUTPUT_C_POLICY):
                    policy[i] = exp_vals[i] / total_exp

        # 提取价值
        radical_factor = adjust_radical_factor(
            SearchParam().max_radical_factor, self.root_game.turn
        )
        # 找到最佳动作的价值
        best_value = ModelOutputValue._get_illegal()
        for result in self.all_action_results:
            if result.is_legal:
                v = result.get_weighted_mean_score(radical_factor)
                if v.value > best_value.value:
                    best_value = v

        # 归一化价值输出
        value = [
            (best_value.score_mean - 38000) / 300,
            best_value.score_stdev / 150,
            (best_value.value - 38000) / 300,
        ]

        return {
            "nn_input": nn_input,
            "policy": policy,
            "value": value,
        }
