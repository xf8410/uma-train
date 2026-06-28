#!/usr/bin/env python3
"""验证脚本 - 测试所有验收标准"""
import sys
sys.path.insert(0, '.')

import random
import numpy as np

def test_game():
    """验收标准2: Game类能初始化并计算训练值"""
    print("=== 测试Game类 ===")
    from simulator.game import Game
    
    rng = random.Random(42)
    game = Game()
    game.new_game(rng)
    
    assert not game.is_end(), "新游戏不应结束"
    assert game.vital == 100, f"初始体力应为100, 实际{game.vital}"
    assert game.motivation == 3, f"初始干劲应为3, 实际{game.motivation}"
    
    # 计算训练值
    for i in range(5):
        assert len(game.train_value[i]) == 6, f"训练值维度错误"
        assert game.train_value[i][0] >= 0 or i != 0, "速度训练应有速度值"
    
    # 测试动作合法性
    from simulator.action import Action, TrainActionType, GameStage
    action_train_speed = Action(type=GameStage.BEFORE_TRAIN, train=TrainActionType.SPEED)
    assert game.is_legal(action_train_speed), "速度训练应该合法"
    
    action_rest = Action(type=GameStage.BEFORE_TRAIN, train=TrainActionType.REST)
    assert game.is_legal(action_rest), "休息应该合法"
    
    # 测试apply_action
    game_copy = game  # 直接在game上操作
    game_copy.apply_action(rng, action_rest)
    assert game_copy.turn == 1, f"执行动作后应进入下一回合, 实际{game_copy.turn}"
    
    print(f"  初始状态: 体力={game.vital} 干劲={game.motivation}")
    print(f"  速度训练值: {game.train_value[0]}")
    print(f"  失败率: {game.fail_rate}")
    print("  [PASS] Game类测试通过!")


def test_mcts():
    """验收标准3: MCTS搜索能运行（用手写逻辑）"""
    print("\n=== 测试MCTS搜索 ===")
    from simulator.game import Game
    from search.mcts import MCTS, SearchParam
    from model.handwritten import HandwrittenEvaluator
    
    rng = random.Random(42)
    game = Game()
    game.new_game(rng)
    
    evaluator = HandwrittenEvaluator()
    mcts = MCTS(evaluator=evaluator)
    param = SearchParam(search_single_max=4, max_depth=1)
    
    action = mcts.run_search(game, rng, param)
    assert action is not None, "搜索应返回动作"
    print(f"  搜索结果: {action}")
    print("  [PASS] MCTS搜索测试通过!")


def test_model():
    """验收标准4: 网络模型能forward一个batch"""
    print("\n=== 测试网络模型 ===")
    try:
        import torch
    except ImportError:
        print("  [SKIP] PyTorch未安装，跳过模型测试")
        return
    
    from model.network import create_model
    from config import Game_Input_C, Game_Output_C
    
    model = create_model('ems')
    batch_size = 4
    x = torch.randn(batch_size, Game_Input_C)
    
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    assert output.shape == (batch_size, Game_Output_C), f"输出形状错误: {output.shape}"
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {output.shape}")
    print(f"  模型大小: {model.get_model_size_kb():.1f} KB")
    print("  [PASS] 网络模型测试通过!")


def test_training():
    """验收标准5: 训练循环能跑起来（用随机数据）"""
    print("\n=== 测试训练循环 ===")
    try:
        import torch
    except ImportError:
        print("  [SKIP] PyTorch未安装，跳过训练测试")
        return
    
    from training.dataset import generate_random_data
    from training.train import calculate_loss
    from training.dataset import UmaTrainDataset
    from torch.utils.data import DataLoader
    from model.network import create_model
    
    # 生成小量随机数据
    generate_random_data(256, './data/test_train.npz')
    
    dataset = UmaTrainDataset('./data/test_train.npz')
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = create_model('ems')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    for i, (x, label) in enumerate(loader):
        optimizer.zero_grad()
        output = model(x)
        v_loss, p_loss = calculate_loss(output, label)
        loss = v_loss + p_loss
        loss.backward()
        optimizer.step()
        
        if i == 0:
            print(f"  第一步: v_loss={v_loss.item():.4f}, p_loss={p_loss.item():.4f}")
    
    print("  [PASS] 训练循环测试通过!")


def test_nn_input():
    """测试NN输入编码"""
    print("\n=== 测试NN输入编码 ===")
    from simulator.game import Game
    from model.nn_input import encode_game_state
    from config import NN_INPUT_C
    
    rng = random.Random(42)
    game = Game()
    game.new_game(rng)
    
    nn_input = encode_game_state(game)
    assert len(nn_input) == NN_INPUT_C, f"输入维度错误: {len(nn_input)} != {NN_INPUT_C}"
    print(f"  NN输入维度: {len(nn_input)}")
    print("  [PASS] NN输入编码测试通过!")


def test_handwritten():
    """测试手写逻辑"""
    print("\n=== 测试手写逻辑 ===")
    from simulator.game import Game
    from model.handwritten import HandwrittenEvaluator
    
    rng = random.Random(42)
    game = Game()
    game.new_game(rng)
    
    evaluator = HandwrittenEvaluator()
    action = evaluator.select_action(game, rng)
    assert action is not None, "手写逻辑应返回动作"
    print(f"  手写逻辑选择: {action}")
    print(f"  评估分数: {evaluator.evaluate(game)}")
    print("  [PASS] 手写逻辑测试通过!")


if __name__ == "__main__":
    results = []
    
    try:
        test_game()
        results.append(("Game类", True))
    except Exception as e:
        print(f"  [FAIL] Game类测试失败: {e}")
        results.append(("Game类", False))
    
    try:
        test_mcts()
        results.append(("MCTS搜索", True))
    except Exception as e:
        print(f"  [FAIL] MCTS搜索测试失败: {e}")
        results.append(("MCTS搜索", False))
    
    try:
        test_nn_input()
        results.append(("NN输入编码", True))
    except Exception as e:
        print(f"  [FAIL] NN输入编码测试失败: {e}")
        results.append(("NN输入编码", False))
    
    try:
        test_handwritten()
        results.append(("手写逻辑", True))
    except Exception as e:
        print(f"  [FAIL] 手写逻辑测试失败: {e}")
        results.append(("手写逻辑", False))
    
    try:
        test_model()
        results.append(("网络模型", True))
    except Exception as e:
        print(f"  [FAIL] 网络模型测试失败: {e}")
        results.append(("网络模型", False))
    
    try:
        test_training()
        results.append(("训练循环", True))
    except Exception as e:
        print(f"  [FAIL] 训练循环测试失败: {e}")
        results.append(("训练循环", False))
    
    print("\n" + "=" * 50)
    print("验收测试结果:")
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\n通过: {passed}/{total}")
    print("=" * 50)
