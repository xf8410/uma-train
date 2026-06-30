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
    HANDWRITTEN_STDEV_BASE, HANDWRITTEN_STDEV_FLOOR,
    COMPARE_WITH_HANDWRITTEN,
    MCTS_BATCH_NN_CALIBRATE,
    VALUE_MEAN_OFFSET, VALUE_MEAN_SCALE, VALUE_STDEV_SCALE,
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
        self.root_snap: Optional[dict] = None  # P0-1: root状态快照，替代deepcopy
        # 决策归因日志：每次run_search清空，搜索结束后可读取
        self.search_log: List[dict] = []

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

        # 每次搜索清空归因日志
        self.search_log = []

        self.root_game = game
        self.root_snap = game.save_snapshot()
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

        # P0-2: 搜索结束后batch NN校准root各动作value
        if MCTS_BATCH_NN_CALIBRATE:
            self._batch_evaluate_root_actions()

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

        # 记录最终选择到归因日志
        self.search_log.append({
            "type": "final_decision",
            "best_action": best_action_int,
            "best_value": best_value,
            "turn": self.root_game.turn,
        })

        # NN vs 手写对账（纯诊断，不影响搜索结果）
        if COMPARE_WITH_HANDWRITTEN:
            self._compare_with_handwritten(self.root_game, rng)

        # 恢复root状态，确保传入的game对象不被搜索修改
        self.root_game.restore_snapshot(self.root_snap)

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
            # 从快照恢复游戏状态（替代deepcopy，快100倍以上）
            self.root_game.restore_snapshot(self.root_snap)
            
            # 应用动作
            self.root_game.apply_action(rng, action)

            # 蒙特卡洛模拟到结束或最大深度
            for depth in range(param.max_depth):
                if self.root_game.is_end():
                    break
                # 选择动作（用策略，返回归因meta但这里不存）
                next_action, _meta = self._select_action_by_policy(self.root_game, rng)
                self.root_game.apply_action(rng, next_action)

            # 评估最终状态
            if self.root_game.is_end():
                score = self.root_game.final_score()
                v = ModelOutputValue(
                    score_mean=float(score),
                    score_stdev=0.0,
                    value=float(score)
                )
            else:
                # 用评估器估算
                v = self._evaluate_game(self.root_game, rng)

            search_result.add_result(v)

    def _select_action_by_policy(self, game: Game, rng: random.Random) -> Tuple[Action, dict]:
        """根据策略选择动作，同时返回归因meta

        如果有神经网络模型，使用网络输出策略；
        否则使用手写逻辑。

        Returns:
            (action, meta) 其中meta包含source、action、top3等归因信息
        """
        if self.model is not None:
            action, meta = self._select_action_by_nn(game, rng)
            return action, meta
        else:
            # 使用手写逻辑
            action = self.evaluator.select_action(game, rng)
            meta = {
                "source": "handwritten",
                "action": action.train if hasattr(action, 'train') else -1,
                "top3_actions_with_probs": [],
            }
            self.search_log.append({
                "type": "select_action",
                **meta,
            })
            return action, meta


    def _select_action_by_nn(self, game: Game, rng: random.Random) -> Tuple[Action, dict]:
        """使用神经网络策略选择动作，记录完整归因信息"""
        import torch
        from model.nn_input import encode_game_state

        # 编码游戏状态
        nn_input = encode_game_state(game)
        nn_input_tensor = torch.FloatTensor(nn_input).unsqueeze(0)

        # 推理
        with torch.no_grad():
            output = self.model(nn_input_tensor)
            policy_logits = output[0, :NN_OUTPUT_C_POLICY]

        # 记录原始logits（只记录合法动作，避免日志过大）
        raw_logits = {}
        for action_int in range(min(NN_OUTPUT_C_POLICY, Action.MAX_ACTION_TYPE)):
            action = Action.from_int(action_int)
            if game.is_legal(action):
                raw_logits[action_int] = round(policy_logits[action_int].item(), 4)

        # 排除不合法动作
        for action_int in range(Action.MAX_ACTION_TYPE):
            action = Action.from_int(action_int)
            if not game.is_legal(action):
                policy_logits[action_int] = -1e7

        # softmax
        probs = torch.softmax(policy_logits, dim=0).numpy()

        # 记录top3
        legal_items = [(i, probs[i]) for i in range(len(probs)) if probs[i] > 1e-8]
        legal_items.sort(key=lambda x: -x[1])
        top3 = [(a, round(p, 6)) for a, p in legal_items[:3]]

        # 按概率采样
        legal_indices = [i for i in range(len(probs)) if probs[i] > 1e-8]
        fallback = False
        if not legal_indices:
            # fallback到手写逻辑
            fallback = True
            action = self.evaluator.select_action(game, rng)
            meta = {
                "source": "fallback",
                "action": action.train if hasattr(action, 'train') else -1,
                "top3_actions_with_probs": top3,
                "raw_logits_sample": raw_logits,
                "softmax_probs_sample": {a: round(p, 6) for a, p in legal_items[:10]},
                "fallback": True,
            }
            self.search_log.append({"type": "select_action", **meta})
            return action, meta

        legal_probs = [probs[i] for i in legal_indices]
        total = sum(legal_probs)
        if total <= 0:
            chosen_idx = rng.choice(legal_indices)
        else:
            legal_probs = [p / total for p in legal_probs]
            chosen_idx = rng.choices(legal_indices, weights=legal_probs, k=1)[0]

        action = Action.from_int(chosen_idx)
        meta = {
            "source": "nn",
            "action": chosen_idx,
            "top3_actions_with_probs": top3,
            "raw_logits_sample": raw_logits,
            "softmax_probs_sample": {a: round(p, 6) for a, p in legal_items[:10]},
            "fallback": False,
        }
        self.search_log.append({"type": "select_action", **meta})
        return action, meta


    def _evaluate_game(self, game: Game, rng: random.Random) -> ModelOutputValue:
        """评估游戏状态，记录归因信息

        如果有神经网络模型，使用模型评估；
        否则使用手写逻辑估算。
        """
        progress = game.turn / TOTAL_TURN
        # 统一使用config中的标准差参数（不再硬编码）
        handwritten_stdev = HANDWRITTEN_STDEV_BASE * (1.0 - progress) + HANDWRITTEN_STDEV_FLOOR

        if self.model is not None:
            import torch
            from model.nn_input import encode_game_state

            nn_input = encode_game_state(game)
            nn_input_tensor = torch.FloatTensor(nn_input).unsqueeze(0)

            with torch.no_grad():
                output = self.model(nn_input_tensor)
                value_output = output[0, NN_OUTPUT_C_POLICY:]

            # 记录NN原始输出
            nn_raw = [round(value_output[i].item(), 6) for i in range(NN_OUTPUT_C_VALUE)]

            # 反归一化
            score_mean = value_output[0].item() * VALUE_MEAN_SCALE + VALUE_MEAN_OFFSET
            score_stdev = value_output[1].item() * VALUE_STDEV_SCALE
            optimistic = value_output[2].item() * VALUE_MEAN_SCALE + VALUE_MEAN_OFFSET

            denormalized_value = ModelOutputValue(
                score_mean=score_mean,
                score_stdev=max(0, score_stdev),
                value=optimistic
            )

            # 同时计算手写评估值（用于对比，不影响搜索结果）
            handwritten_score = self.evaluator.evaluate(game)

            self.search_log.append({
                "type": "evaluate_game",
                "source": "nn",
                "nn_raw_output": nn_raw,
                "denormalized_value": {
                    "score_mean": round(denormalized_value.score_mean, 2),
                    "score_stdev": round(denormalized_value.score_stdev, 2),
                    "value": round(denormalized_value.value, 2),
                },
                "handwritten_value": handwritten_score,
                "turn": game.turn,
            })

            return denormalized_value
        else:
            # 手写逻辑评估
            score = self.evaluator.evaluate(game)
            result = ModelOutputValue(
                score_mean=float(score),
                score_stdev=handwritten_stdev,
                value=float(score)
            )

            self.search_log.append({
                "type": "evaluate_game",
                "source": "handwritten",
                "nn_raw_output": None,
                "denormalized_value": {
                    "score_mean": round(result.score_mean, 2),
                    "score_stdev": round(result.score_stdev, 2),
                    "value": round(result.value, 2),
                },
                "handwritten_value": score,
                "turn": game.turn,
            })

            return result


    def _compare_with_handwritten(self, game: Game, rng: random.Random):
        """NN vs 手写对账：对root_game的每个合法action同时跑两种评估

        纯诊断用，不影响搜索结果，记录到search_log。
        """
        comparison_results = []

        for action_int in range(Action.MAX_ACTION_TYPE):
            action = Action.from_int(action_int)
            if not game.is_legal(action):
                continue

            # 手写评估
            hw_value = self.evaluator._evaluate_action(game, action, rng)

            # NN评估（如果模型可用）
            nn_value = None
            if self.model is not None:
                try:
                    import torch
                    from model.nn_input import encode_game_state
                    game.restore_snapshot(self.root_snap)
                    game.apply_action(rng, action)
                    nn_input = encode_game_state(game)
                    nn_input_tensor = torch.FloatTensor(nn_input).unsqueeze(0)
                    with torch.no_grad():
                        output = self.model(nn_input_tensor)
                        value_output = output[0, NN_OUTPUT_C_POLICY:]
                    nn_value = round(value_output[0].item() * VALUE_MEAN_SCALE + VALUE_MEAN_OFFSET, 2)
                except Exception:
                    nn_value = None

            comparison_results.append({
                "action": action_int,
                "handwritten_value": round(hw_value, 2),
                "nn_value": nn_value,
                "diff": round(nn_value - hw_value, 2) if nn_value is not None else None,
            })

        self.search_log.append({
            "type": "nn_vs_handwritten_comparison",
            "comparisons": comparison_results,
        })


    def _batch_evaluate_root_actions(self):
        """搜索结束后batch NN校准root各动作value（P0-2）

        对root的每个合法动作推进一步，用batch NN推理评估后续状态，
        将NN评估结果作为额外样本加入各动作的SearchResult，
        提升搜索结果的准确性。
        """
        if self.model is None:
            return

        import torch
        from model.nn_input import encode_game_state

        # 收集合法动作
        legal_actions = []
        for action_int in range(Action.MAX_ACTION_TYPE):
            if self.all_action_results[action_int].is_legal:
                legal_actions.append(action_int)
        if not legal_actions:
            return

        # 逐动作推进一步，编码状态
        nn_inputs = []
        root_snap = self.root_game.save_snapshot()
        for action_int in legal_actions:
            self.root_game.restore_snapshot(root_snap)
            action = Action.from_int(action_int)
            self.root_game.apply_action(random.Random(0), action)
            nn_inputs.append(encode_game_state(self.root_game))
        # 恢复root状态
        self.root_game.restore_snapshot(root_snap)

        # batch推理
        nn_input_tensor = torch.FloatTensor(nn_inputs)
        with torch.no_grad():
            output = self.model(nn_input_tensor)
            value_output = output[:, NN_OUTPUT_C_POLICY:]

        # 用NN结果校准各动作的value
        for i, action_int in enumerate(legal_actions):
            score_mean = value_output[i, 0].item() * VALUE_MEAN_SCALE + VALUE_MEAN_OFFSET
            score_stdev = max(0, value_output[i, 1].item() * VALUE_STDEV_SCALE)
            optimistic = value_output[i, 2].item() * VALUE_MEAN_SCALE + VALUE_MEAN_OFFSET
            nn_v = ModelOutputValue(score_mean=score_mean, score_stdev=score_stdev, value=optimistic)
            self.all_action_results[action_int].add_result(nn_v)

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
            (best_value.score_mean - VALUE_MEAN_OFFSET) / VALUE_MEAN_SCALE,
            best_value.score_stdev / VALUE_STDEV_SCALE,
            (best_value.value - VALUE_MEAN_OFFSET) / VALUE_MEAN_SCALE,
        ]

        return {
            "nn_input": nn_input,
            "policy": policy,
            "value": value,
        }
