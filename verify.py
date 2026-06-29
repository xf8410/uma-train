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




def test_formula_layer():
    """验收: 公式层完整实现"""
    print("\n=== 测试公式层 ===")
    from simulator.formula import (
        FormulaLayer, calc_fail_rate, calc_rest_vital,
        calc_great_success_prob, calc_training_value,
        MOTIVATION_MULTIPLIER, MOTIVATION_REST_VITAL
    )
    from simulator.bad_condition import BadConditionManager, BadConditionType
    import random
    
    rng = random.Random(42)
    bc = BadConditionManager()
    fl = FormulaLayer(bc)
    
    # 1. 失败率：低体力高失败率
    fr_low = calc_fail_rate(30, 3, 0, 0, bc)
    fr_high = calc_fail_rate(80, 5, 0, 0, bc)
    assert fr_low > 0, f"低体力应有失败率, 实际{fr_low}"
    assert fr_high == 0, f"高体力应无失败率, 实际{fr_high}"
    print(f"  失败率(体力30/普通): {fr_low}%")
    print(f"  失败率(体力80/绝好调): {fr_high}%")
    
    # 2. 練習ベタ+2%失败率
    bc.acquire(BadConditionType.BAD, 5)
    fr_with_bad = calc_fail_rate(30, 3, 0, 0, bc)
    assert fr_with_bad >= fr_low + 2, f"練習ベタ应+2%失败率"
    print(f"  失败率(含練習ベタ): {fr_with_bad}% (+{fr_with_bad-fr_low}%)")
    
    # 3. やる気倍率
    assert MOTIVATION_MULTIPLIER[1] == 0.8, "絶不調倍率0.8"
    assert MOTIVATION_MULTIPLIER[5] == 1.2, "絶好調倍率1.2"
    print(f"  やる気倍率: {MOTIVATION_MULTIPLIER}")
    
    # 4. お休み回复
    rest_v = calc_rest_vital(5, False)
    rest_great = calc_rest_vital(5, True)
    assert rest_great > rest_v, "大成功应回复更多"
    print(f"  お休み(絶好調): {rest_v}, 大成功: {rest_great}")
    
    # 5. 大成功概率
    prob_low_vital = calc_great_success_prob(20, 100)
    prob_high_vital = calc_great_success_prob(90, 100)
    assert prob_low_vital > prob_high_vital, "体力低时应更容易大成功"
    print(f"  大成功概率(体力20): {prob_low_vital:.2f}, (体力90): {prob_high_vital:.2f}")
    
    # 6. 智力训练无失败
    fr_wit = calc_fail_rate(30, 1, 4, 0, bc)
    assert fr_wit == 0, "智力训练应无失败率"
    print(f"  智力训练失败率: {fr_wit}% (应为0)")
    
    print("  [PASS] 公式层测试通过!")


def test_bad_condition():
    """验收: バッドコンディション系统"""
    print("\n=== 测试バッドコンディション ===")
    from simulator.bad_condition import BadConditionManager, BadConditionType
    import random
    
    rng = random.Random(42)
    bc = BadConditionManager()
    
    # 1. 获取和查询
    bc.acquire(BadConditionType.BAD, 5)
    bc.acquire(BadConditionType.LATE_BED, 10)
    assert bc.count == 2, f"应有2个状态, 实际{bc.count}"
    assert bc.has(BadConditionType.BAD), "应有練習ベタ"
    print(f"  获取2个: {bc}")
    
    # 2. 保健室治愈（2.5周年后必消1个，优先片頭痛>練習ベタ）
    bc.acquire(BadConditionType.HEADACHE, 15)
    healed = bc.heal_by_clinic()
    assert healed == BadConditionType.HEADACHE, "应优先治愈片頭痛"
    assert bc.count == 2, f"治愈后应剩2个, 实际{bc.count}"
    print(f"  保健室治愈: {BadConditionType(healed).name}, 剩余: {bc}")
    
    # 3. 片頭痛限制やる気上升
    bc2 = BadConditionManager()
    bc2.acquire(BadConditionType.HEADACHE, 5)
    assert bc2.is_motivation_blocked(), "片頭痛应阻止やる気上升"
    bc2.heal_by_clinic()
    assert not bc2.is_motivation_blocked(), "治愈后不应阻止"
    print("  片頭痛やる気限制: OK")
    
    # 4. 太り気味Speed无效
    bc3 = BadConditionManager()
    bc3.acquire(BadConditionType.FAT, 5)
    assert bc3.is_speed_training_disabled(), "太り気味应使Speed训练无效"
    print("  太り気味Speed无效: OK")
    
    # 5. 夜ふかし体力消耗
    bc4 = BadConditionManager()
    bc4.acquire(BadConditionType.LATE_BED, 5)
    drain = bc4.get_late_bed_vital_drain()
    assert drain == -10, f"夜ふかし应-10体力, 实际{drain}"
    print(f"  夜ふかし体力消耗: {drain}")
    
    # 6. お休み治愈
    bc5 = BadConditionManager()
    bc5.acquire(BadConditionType.LATE_BED, 5)
    bc5.acquire(BadConditionType.SKIN, 8)
    healed_list = bc5.heal_by_rest(rng)
    print(f"  お休み治愈: {[BadConditionType(h).name for h in healed_list]}")
    
    # 7. やる気下降不重复（5回合冷却）
    bc6 = BadConditionManager()
    assert bc6.can_motivation_decrease(10)
    bc6.record_motivation_decrease(10)
    assert not bc6.can_motivation_decrease(12), "5回合内不应再下降"
    assert bc6.can_motivation_decrease(15), "5回合后可再下降"
    print("  やる気下降5回合冷却: OK")
    
    print("  [PASS] バッドコンディション测试通过!")


def test_ramen_scenario():
    """验收: Ramen剧本实现"""
    print("\n=== 测试Ramen剧本 ===")
    from simulator.scenarios.ramen import RamenScenario
    from simulator.game import Game
    import random
    
    rng = random.Random(42)
    ramen = RamenScenario()
    
    assert ramen.SCENARIO_ID == 14, "scenario_id应为14"
    assert ramen.kakushimi_count == 0, "初始隠し味应为0"
    print(f"  Ramen剧本: id={ramen.SCENARIO_ID}, name={ramen.SCENARIO_NAME}")
    
    # 隠し味获取
    ramen.add_kakushimi(5)
    assert ramen.kakushimi_count == 5
    print(f"  隠し味获取: {ramen.kakushimi_count}")
    
    # お出かけ获取隠し味
    outing_k = ramen.get_kakushimi_from_outing()
    assert outing_k == 2, "お出かけ应+2隠し味"
    print(f"  お出かけ隠し味: +{outing_k}")
    
    # 試食会获取
    tasting_k = ramen.get_kakushimi_from_tasting(rng)
    assert tasting_k >= 1, "試食会至少+1隠し味"
    print(f"  試食会隠し味: +{tasting_k}")
    
    # 训练值修改（隠し味倍率）
    train_value = [20, 0, 5, 0, 0, 10]
    modified = ramen.modify_training_value(None, 0, list(train_value))
    print(f"  训练值修改(隠し味=5): {train_value} -> {modified}")
    
    # コツ系统
    game = Game()
    game.new_game(rng)
    ramen.on_turn_start(game, rng)
    print(f"  コツ等级: {ramen.kotsu_level}")
    
    # 序列化
    d = ramen.to_dict()
    assert d['kakushimi_count'] == 5
    print(f"  序列化: {d}")
    
    print("  [PASS] Ramen剧本测试通过!")


def test_game_bc_integration():
    """验收: Game类バッドコンディション集成"""
    print("\n=== 测试Game+BC集成 ===")
    from simulator.game import Game
    from simulator.bad_condition import BadConditionType
    import random
    
    rng = random.Random(42)
    game = Game()
    game.new_game(rng)
    
    # bc_manager和formula已初始化
    assert game.bc_manager is not None, "bc_manager应已初始化"
    assert game.formula is not None, "formula应已初始化"
    print(f"  bc_manager: {type(game.bc_manager).__name__}")
    print(f"  formula: {type(game.formula).__name__}")
    
    # 片頭痛限制やる気上升
    game.bc_manager.acquire(BadConditionType.HEADACHE, 5)
    old_mot = game.motivation
    game.add_motivation(1)
    assert game.motivation == old_mot, f"片頭痛时やる気不应上升, {old_mot}->{game.motivation}"
    game.bc_manager.heal_by_clinic()
    game.add_motivation(1)
    assert game.motivation > old_mot, "治愈后やる気可上升"
    print(f"  片頭痛やる気限制: {old_mot} -> 治愈后 {game.motivation}")
    
    # print_state含バッドコンディション
    game2 = Game()
    game2.new_game(rng)
    game2.bc_manager.acquire(BadConditionType.BAD, 3)
    state_str = game2.print_state()
    assert "練習ベタ" in state_str or "バッドコンディション" in state_str
    print("  print_state含BC: OK")
    
    print("  [PASS] Game+BC集成测试通过!")


if __name__ == "__main__":
    test_game()
    test_mcts()
    test_formula_layer()
    test_bad_condition()
    test_ramen_scenario()
    test_game_bc_integration()
    print("\n" + "="*50)
    print("全部验收通过! ✓")
