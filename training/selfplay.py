"""
赛马娘AI训练框架 - 自我对弈

生成MCTS搜索训练数据，用于神经网络训练。
"""

import random
import numpy as np
from copy import deepcopy
from typing import List, Optional

from simulator.game import Game
from search.mcts import MCTS, SearchParam
from search.search_result import ModelOutputValue
from model.nn_input import encode_game_state
from model.handwritten import HandwrittenEvaluator
from config import TOTAL_TURN, NN_INPUT_C, NN_OUTPUT_C_POLICY, NN_OUTPUT_C_VALUE


class SelfPlayWorker:
    """自我对弈工作者
    
    使用MCTS搜索+手写逻辑/神经网络生成训练数据。
    """
    
    def __init__(
        self,
        model=None,               # 神经网络模型（可选）
        search_param: Optional[SearchParam] = None,
        evaluator: Optional[HandwrittenEvaluator] = None,
    ):
        """
        Args:
            model: PyTorch模型，为None时使用手写逻辑
            search_param: 搜索参数
            evaluator: 手写评估器
        """
        self.model = model
        self.search_param = search_param or SearchParam(
            search_single_max=64,  # 自我对弈时减少搜索量
            max_depth=5,
        )
        self.evaluator = evaluator or HandwrittenEvaluator()
        self.mcts = MCTS(model=model, evaluator=self.evaluator)
    
    def play_game(
        self,
        rng: random.Random,
        uma_id: int = 0,
        card_ids: Optional[List[int]] = None,
        zhong_ma_blue: Optional[List[int]] = None,
        card_db: Optional[dict] = None,
    ) -> List[dict]:
        """进行一局自我对弈
        
        Args:
            rng: 随机数生成器
            uma_id: 马娘ID
            card_ids: 支援卡ID列表
            zhong_ma_blue: 种马蓝因子
            card_db: 支援卡数据库
            
        Returns:
            训练样本列表
        """
        game = Game()
        game.new_game(rng, uma_id=uma_id, card_ids=card_ids,
                      zhong_ma_blue=zhong_ma_blue, card_db=card_db)
        
        samples = []
        
        while not game.is_end():
            # 编码当前状态
            nn_input = encode_game_state(game)
            
            # MCTS搜索
            action = self.mcts.run_search(game, rng, self.search_param)
            
            # 提取策略
            policy = self._extract_policy(game)
            
            # 记录样本
            samples.append({
                "nn_input": nn_input,
                "policy": policy,
                "turn": game.turn,
            })
            
            # 应用动作
            game.apply_action(rng, action)
        
        # 游戏结束，计算最终得分
        final_score = game.final_score()
        
        # 回填价值
        for sample in samples:
            turn = sample["turn"]
            # 根据剩余回合数估算标准差
            progress = turn / TOTAL_TURN
            stdev = 500.0 * (1.0 - progress) + 100.0
            
            sample["value"] = [
                (final_score - 38000) / 300,
                stdev / 150,
                (final_score - 38000) / 300,  # 乐观分 = 平均分
            ]
            sample["final_score"] = final_score
        
        return samples
    
    def _extract_policy(self, game: Game) -> List[float]:
        """从MCTS搜索结果提取策略"""
        policy = [0.0] * NN_OUTPUT_C_POLICY
        total_visit = 0
        
        for action_int in range(min(NN_OUTPUT_C_POLICY, len(self.mcts.all_action_results))):
            result = self.mcts.all_action_results[action_int]
            if result.is_legal:
                policy[action_int] = result.num
                total_visit += result.num
        
        # 归一化
        if total_visit > 0:
            for i in range(NN_OUTPUT_C_POLICY):
                policy[i] /= total_visit
        
        return policy
    
    def generate_batch(
        self,
        num_games: int,
        rng: Optional[random.Random] = None,
    ) -> tuple:
        """批量生成自我对弈数据
        
        Args:
            num_games: 游戏局数
            rng: 随机数生成器
            
        Returns:
            (x, label) numpy数组
        """
        if rng is None:
            rng = random.Random()
        
        all_nn_inputs = []
        all_policies = []
        all_values = []
        
        for game_idx in range(num_games):
            print(f"  自我对弈第 {game_idx + 1}/{num_games} 局...")
            samples = self.play_game(rng)
            
            for sample in samples:
                all_nn_inputs.append(sample["nn_input"])
                all_policies.append(sample["policy"])
                all_values.append(sample["value"])
        
        if not all_nn_inputs:
            return np.zeros((0, NN_INPUT_C)), np.zeros((0, NN_OUTPUT_C_POLICY + NN_OUTPUT_C_VALUE))
        
        x = np.array(all_nn_inputs, dtype=np.float32)
        policy = np.array(all_policies, dtype=np.float32)
        value = np.array(all_values, dtype=np.float32)
        label = np.concatenate([policy, value], axis=1)
        
        return x, label
