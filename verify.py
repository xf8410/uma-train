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
    from config import NN_INPUT_C, NN_OUTPUT_C
    
    model = create_model('ems')
    batch_size = 4
    x = torch.randn(batch_size, NN_INPUT_C)
    
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    assert output.shape == (batch_size, NN_OUTPUT_C), f"输出形状错误: {output.shape}"
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



# ============================================================
# 新增测试组：MCTS/NN/训练循环覆盖扩展
# ============================================================

def test_mcts_full_game():
    print(chr(10) + '=== 测试MCTS完整搜索 ===')
    from simulator.game import Game
    from search.mcts import MCTS, SearchParam
    from model.handwritten import HandwrittenEvaluator
    from simulator.action import Action, TrainActionType, GameStage

    rng = random.Random(123)
    game = Game()
    game.new_game(rng)

    evaluator = HandwrittenEvaluator()
    mcts = MCTS(evaluator=evaluator)
    param = SearchParam(search_single_max=4, max_depth=1, search_group_size=2)

    turn_count = 0
    while not game.is_end() and turn_count < 10:
        action = mcts.run_search(game, rng, param)
        assert action is not None, '搜索返回None'
        assert game.is_legal(action), '搜索返回了不合法动作'
        game.apply_action(rng, action)
        turn_count += 1

    print(f'  成功跑过 {turn_count} 回合，无crash')

    mcts2 = MCTS(evaluator=evaluator)
    game2 = Game()
    game2.new_game(rng)
    action2 = mcts2.run_search(game2, rng, param)
    assert action2 is not None, '搜索应返回动作'
    legal_searched = sum(1 for r in mcts2.all_action_results if r.is_legal and r.num > 0)
    assert legal_searched > 0, '搜索结果中应至少有一个合法动作被选中'
    print(f'  搜索结果中有 {legal_searched} 个合法动作被选中')

    try:
        mcts2.print_search_result(param, show_search_num=True)
        print('  search_log可读: OK')
    except Exception as e:
        assert False, f'print_search_result失败: {e}'

    print('  [PASS] MCTS完整搜索测试通过!')


def test_handwritten_evaluator():
    print(chr(10) + '=== 测试手写评估器 ===')
    from simulator.game import Game
    from model.handwritten import HandwrittenEvaluator
    from simulator.bad_condition import BadConditionType
    from simulator.scenarios.ramen import RamenScenario
    from simulator.action import Action, TrainActionType, GameStage

    rng = random.Random(42)
    game = Game()
    game.new_game(rng)
    evaluator = HandwrittenEvaluator()

    score_base = evaluator.evaluate(game)
    assert isinstance(score_base, int)
    print(f'  基础评估值: {score_base}')

    for _ in range(5):
        action = evaluator.select_action(game, rng)
        game.apply_action(rng, action)
    score_mid = evaluator.evaluate(game)
    print(f'  中期评估值: {score_mid}')

    game_bc = Game()
    game_bc.new_game(rng)
    val_no_bc = evaluator._evaluate_action(game_bc,
        Action(type=GameStage.BEFORE_TRAIN, train=TrainActionType.SPEED), rng)
    game_bc.bc_manager.acquire(BadConditionType.BAD, 3)
    val_with_bc = evaluator._evaluate_action(game_bc,
        Action(type=GameStage.BEFORE_TRAIN, train=TrainActionType.SPEED), rng)
    assert val_with_bc < val_no_bc
    print(f'  BC惩罚生效: 无BC={val_no_bc:.1f}, 有BC={val_with_bc:.1f} (差={val_no_bc - val_with_bc:.1f})')

    game_ramen = Game()
    game_ramen.new_game(rng)
    val_no_ramen = evaluator._evaluate_action(game_ramen,
        Action(type=GameStage.BEFORE_TRAIN, train=TrainActionType.SPEED), rng)
    ramen = RamenScenario()
    ramen.add_kakushimi(20)
    game_ramen.scenario = ramen
    val_with_ramen = evaluator._evaluate_action(game_ramen,
        Action(type=GameStage.BEFORE_TRAIN, train=TrainActionType.SPEED), rng)
    assert val_with_ramen > val_no_ramen
    print(f'  Ramen加成生效: 无Ramen={val_no_ramen:.1f}, 有Ramen={val_with_ramen:.1f} (加={val_with_ramen - val_no_ramen:.1f})')

    print('  [PASS] 手写评估器测试通过!')


def test_nn_input_encoding():
    print(chr(10) + '=== 测试NN输入编码 ===')
    from simulator.game import Game
    from model.nn_input import encode_game_state
    from config import NN_INPUT_C
    from simulator.action import Action, TrainActionType, GameStage

    rng = random.Random(42)

    game = Game()
    game.new_game(rng)
    nn_input = encode_game_state(game)
    assert len(nn_input) == NN_INPUT_C
    print(f'  编码长度: {len(nn_input)} (期望 {NN_INPUT_C})')

    min_val = min(nn_input)
    max_val = max(nn_input)
    assert min_val >= -2.0
    assert max_val <= 2.0
    print(f'  值范围: [{min_val:.4f}, {max_val:.4f}]')

    game2 = Game()
    game2.new_game(rng)
    for _ in range(5):
        act = Action(type=GameStage.BEFORE_TRAIN, train=TrainActionType.SPEED)
        if game2.is_legal(act):
            game2.apply_action(rng, act)
        else:
            act = Action(type=GameStage.BEFORE_TRAIN, train=TrainActionType.REST)
            game2.apply_action(rng, act)
    nn_input2 = encode_game_state(game2)

    diff_count = sum(1 for a, b in zip(nn_input, nn_input2) if abs(a - b) > 1e-6)
    assert diff_count > 0
    total_diff = sum(abs(a - b) for a, b in zip(nn_input, nn_input2))
    print(f'  编码差异: {diff_count}/{NN_INPUT_C}个维度不同, 总差={total_diff:.2f}')

    print('  [PASS] NN输入编码测试通过!')


def test_training_loss():
    print(chr(10) + '=== 测试训练损失计算 ===')
    try:
        import torch
    except ImportError:
        print('  [SKIP] PyTorch未安装，跳过训练损失测试')
        return

    from training.train import calculate_loss
    from config import NN_OUTPUT_C, NN_OUTPUT_C_POLICY, NN_OUTPUT_C_VALUE

    batch_size = 8
    output = torch.randn(batch_size, NN_OUTPUT_C)
    label_policy = torch.nn.functional.softmax(torch.randn(batch_size, NN_OUTPUT_C_POLICY), dim=1)
    label_value = torch.randn(batch_size, NN_OUTPUT_C_VALUE) * 100 + 38000
    label = torch.cat([label_policy, label_value], dim=1)

    v_loss, p_loss = calculate_loss(output, label)

    assert torch.isfinite(v_loss)
    assert torch.isfinite(p_loss)
    print(f'  value_loss: {v_loss.item():.4f}')
    print(f'  policy_loss: {p_loss.item():.4f}')

    assert v_loss.item() >= 0
    assert p_loss.item() >= 0
    print('  两个loss均为非负有限值: OK')

    total_loss = v_loss + p_loss
    total_loss.backward()
    print('  反向传播正常: OK')

    print('  [PASS] 训练损失计算测试通过!')



# ============================================================
# P2-1新增测试组：搜索结果合理性、自我对弈标签、完整流程、Welford统计、config拆分
# ============================================================

def test_mcts_search_result_sanity():
    """P2-1新增: MCTS搜索结果合理性 — value范围、策略概率和、合法动作数"""
    print('\n=== 测试MCTS搜索结果合理性 ===')
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

    # 1. 合法动作数 > 0
    legal_count = sum(1 for r in mcts.all_action_results if r.is_legal)
    assert legal_count > 0, f'合法动作数应>0, 实际{legal_count}'
    print(f'  合法动作数: {legal_count}')

    # 2. 搜索结果的value值在合理范围（0到MAX_SCORE）
    from config import MAX_SCORE
    for i, r in enumerate(mcts.all_action_results):
        if not r.is_legal or r._count <= 0:
            continue
        val = r.get_weighted_mean_score(1.0)
        assert 0 <= val.score_mean <= MAX_SCORE, \
            f'动作{i}的score_mean={val.score_mean}超出合理范围[0,{MAX_SCORE}]'
        assert val.score_stdev >= 0, \
            f'动作{i}的score_stdev={val.score_stdev}应为非负'
    print(f'  所有合法动作value在合理范围: OK')

    # 3. 导出训练样本的policy概率和为1
    sample = mcts.export_training_sample()
    policy_sum = sum(sample['policy'])
    assert abs(policy_sum - 1.0) < 1e-4, f'策略概率和应为1, 实际{policy_sum}'
    print(f'  策略概率和: {policy_sum:.6f} (接近1)')

    # 4. value字段长度为3且值合理
    value = sample['value']
    assert len(value) == 3, f'value维度应为3, 实际{len(value)}'
    assert -200 < value[0] < 200, f'value[0]={value[0]}超出合理范围'
    assert value[1] >= 0, f'stdev应为非负, 实际{value[1]}'
    assert -200 < value[2] < 200, f'value[2]={value[2]}超出合理范围'
    print(f'  value字段: mean={value[0]:.2f}, stdev={value[1]:.2f}, optimistic={value[2]:.2f}')

    print('  [PASS] MCTS搜索结果合理性测试通过!')


def test_selfplay_value_labels():
    """P2-1新增: 自我对弈value标签 — sample中value/policy/stdev字段存在且合理"""
    print('\n=== 测试自我对弈value标签 ===')
    import importlib.util
    import random
    from simulator.game import Game
    from search.mcts import MCTS, SearchParam
    from search.search_result import ModelOutputValue
    from model.nn_input import encode_game_state
    from model.handwritten import HandwrittenEvaluator
    from config import TOTAL_TURN, NN_INPUT_C, NN_OUTPUT_C_POLICY, NN_OUTPUT_C_VALUE, HANDWRITTEN_STDEV_BASE, HANDWRITTEN_STDEV_FLOOR

    # 直接加载selfplay模块，避免train.py的torch依赖
    spec = importlib.util.spec_from_file_location('selfplay', './training/selfplay.py')
    sp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sp)

    rng = random.Random(42)
    evaluator = HandwrittenEvaluator()
    param = SearchParam(search_single_max=4, max_depth=1)
    worker = sp.SelfPlayWorker(search_param=param, evaluator=evaluator)
    samples = worker.play_game(rng)

    assert len(samples) > 0, '应生成至少1个样本'
    print(f'  生成样本数: {len(samples)}')

    # 检查关键字段
    for i, s in enumerate(samples):
        # value字段存在且合理
        assert 'value' in s, f'样本{i}缺少value字段'
        v = s['value']
        assert len(v) == 3, f'样本{i}的value维度应为3, 实际{len(v)}'
        assert v[1] >= 0, f'样本{i}的stdev={v[1]}应为非负'

        # policy字段存在且概率和约等于1
        assert 'policy' in s, f'样本{i}缺少policy字段'
        p = s['policy']
        assert len(p) == NN_OUTPUT_C_POLICY, f'样本{i}的policy维度应为{NN_OUTPUT_C_POLICY}'
        psum = sum(p)
        assert abs(psum - 1.0) < 0.01, f'样本{i}的策略概率和={psum}不接近1'

        # stdev字段（mcts_search_stdev）存在
        assert 'mcts_search_stdev' in s, f'样本{i}缺少mcts_search_stdev字段'

    print(f'  所有{len(samples)}个样本value/policy/stdev字段完整且合理')

    # 最终评分>0
    final = samples[0].get('final_score', 0)
    assert final > 0, f'最终评分应>0, 实际{final}'
    print(f'  最终评分: {final} (>0)')

    print('  [PASS] 自我对弈value标签测试通过!')


def test_full_game_78turns():
    """P2-1新增: 多回合完整流程 — 跑完整78回合育成，确认不crash且最终评分>0"""
    print('\n=== 测试完整78回合育成 ===')
    try:
        import torch
        has_torch = True
    except ImportError:
        has_torch = False

    if not has_torch:
        print('  [INFO] 无PyTorch，使用手写逻辑跑完整育成（可能较慢）')

    import random
    from simulator.game import Game
    from search.mcts import MCTS, SearchParam
    from model.handwritten import HandwrittenEvaluator
    from config import TOTAL_TURN

    rng = random.Random(99)
    game = Game()
    game.new_game(rng)

    evaluator = HandwrittenEvaluator()
    # 用极少搜索量加快速度
    mcts = MCTS(evaluator=evaluator)
    param = SearchParam(search_single_max=2, max_depth=1, search_group_size=1)

    turn_count = 0
    while not game.is_end():
        action = mcts.run_search(game, rng, param)
        assert action is not None, f'第{turn_count}回合搜索返回None'
        assert game.is_legal(action), f'第{turn_count}回合搜索返回不合法动作'
        game.apply_action(rng, action)
        turn_count += 1

    assert game.turn == TOTAL_TURN, f'游戏回合数应为{TOTAL_TURN}, 实际{game.turn}'
    score = game.final_score()
    assert score > 0, f'最终评分应>0, 实际{score}'
    print(f'  完整{turn_count}回合育成: 评分={score} (>0), 无crash')

    print('  [PASS] 完整78回合育成测试通过!')


def test_search_result_welford():
    """P2-1新增: SearchResult Welford统计 — 添加少量结果后检查mean/stdev/count"""
    print('\n=== 测试SearchResult Welford统计 ===')
    import math
    from search.search_result import SearchResult, ModelOutputValue
    from config import NORM_DISTRIBUTION_SAMPLING, MAX_SCORE

    sr = SearchResult()
    sr.is_legal = True

    # 添加第一个结果: mean=40000, stdev=200
    v1 = ModelOutputValue(score_mean=40000.0, score_stdev=200.0, value=40000.0)
    sr.add_result(v1)
    n = NORM_DISTRIBUTION_SAMPLING  # 128
    assert sr._count == n, f'添加1次后count应为{n}, 实际{sr._count}'
    assert abs(sr._mean - 40000.0) < 1e-6, f'mean应为40000, 实际{sr._mean}'
    expected_m2 = n * 200.0 * 200.0
    assert abs(sr._m2 - expected_m2) < 1e-6, f'm2应为{expected_m2}, 实际{sr._m2}'
    print(f'  添加1个结果: count={sr._count}, mean={sr._mean:.1f}, m2={sr._m2:.1f}')

    # 添加第二个结果: mean=38000, stdev=300
    v2 = ModelOutputValue(score_mean=38000.0, score_stdev=300.0, value=38000.0)
    sr.add_result(v2)
    n2 = n
    n_combined = n + n2  # 256
    expected_mean = 40000.0 + n2 * (38000.0 - 40000.0) / n_combined  # 39000
    m2_1 = n * 200.0**2
    m2_2 = n2 * 300.0**2
    delta = 38000.0 - 40000.0
    expected_m2 = m2_1 + m2_2 + n * n2 * delta**2 / n_combined
    assert sr._count == n_combined, f'count应为{n_combined}, 实际{sr._count}'
    assert abs(sr._mean - expected_mean) < 1e-6, f'mean应为{expected_mean}, 实际{sr._mean}'
    assert abs(sr._m2 - expected_m2) < 1e-3, f'm2应为{expected_m2}, 实际{sr._m2}'
    expected_stdev = math.sqrt(expected_m2 / n_combined)
    val = sr.get_weighted_mean_score(1.0)
    assert abs(val.score_stdev - expected_stdev) < 0.01, \
        f'stdev应为{expected_stdev:.2f}, 实际{val.score_stdev:.2f}'
    print(f'  添加2个结果: count={sr._count}, mean={sr._mean:.1f}, stdev={val.score_stdev:.2f}')

    # 验证min/max
    lo1 = max(0, 40000 - 3.5 * 200)
    hi1 = min(MAX_SCORE - 1, 40000 + 3.5 * 200)
    lo2 = max(0, 38000 - 3.5 * 300)
    hi2 = min(MAX_SCORE - 1, 38000 + 3.5 * 300)
    assert abs(sr._min - min(lo1, lo2)) < 1e-6, f'min应为{min(lo1,lo2)}, 实际{sr._min}'
    assert abs(sr._max - max(hi1, hi2)) < 1e-6, f'max应为{max(hi1,hi2)}, 实际{sr._max}'
    print(f'  min={sr._min:.1f}, max={sr._max:.1f}')

    # 添加第三个结果验证继续正确
    v3 = ModelOutputValue(score_mean=42000.0, score_stdev=100.0, value=42000.0)
    sr.add_result(v3)
    n3 = n
    n_total = n_combined + n3  # 384
    expected_mean3 = expected_mean + n3 * (42000.0 - expected_mean) / n_total
    delta3 = 42000.0 - expected_mean
    m2_3 = n3 * 100.0**2
    expected_m2_3 = expected_m2 + m2_3 + n_combined * n3 * delta3**2 / n_total
    assert abs(sr._mean - expected_mean3) < 1e-4, f'3次后mean应为{expected_mean3}, 实际{sr._mean}'
    assert abs(sr._m2 - expected_m2_3) < 1e-2, f'3次后m2应为{expected_m2_3}, 实际{sr._m2}'
    print(f'  添加3个结果: count={sr._count}, mean={sr._mean:.2f}, m2={sr._m2:.2f}')

    print('  [PASS] SearchResult Welford统计测试通过!')


def test_config_split():
    """P2-1新增: config拆分验证 — from config import *和from config_nn import都能工作"""
    print('\n=== 测试config拆分验证 ===')

    # 1. from config import * 应能拿到所有常量
    from config import (
        TOTAL_TURN, MAX_SCORE, NN_INPUT_C, NN_OUTPUT_C, NN_OUTPUT_C_POLICY, NN_OUTPUT_C_VALUE,
        HANDWRITTEN_STDEV_BASE, HANDWRITTEN_STDEV_FLOOR, NORM_DISTRIBUTION_SAMPLING,
        SEARCH_SINGLE_MAX, MCTS_POLICY_TEMPERATURE, LEARNING_RATE, BATCH_SIZE,
        SCORE_PT_RATE_DEFAULT, FAIL_RATE_FORMULA_DENOM,
        MODEL_SIZE_LIMIT, DEFAULT_ENCODER_BLOCKS, VALUE_MEAN_OFFSET, VALUE_MEAN_SCALE,
    )
    assert TOTAL_TURN == 78, f'TOTAL_TURN应为78, 实际{TOTAL_TURN}'
    assert NN_INPUT_C > 0, f'NN_INPUT_C应>0'
    assert NN_OUTPUT_C_POLICY > 0, f'NN_OUTPUT_C_POLICY应>0'
    assert NN_OUTPUT_C_VALUE > 0, f'NN_OUTPUT_C_VALUE应>0'
    print(f'  config.* 可用: TOTAL_TURN={TOTAL_TURN}, NN_INPUT_C={NN_INPUT_C}, '
          f'NN_OUTPUT_C={NN_OUTPUT_C}, LEARNING_RATE={LEARNING_RATE}')

    # 2. from config_nn import NN_INPUT_C 也能工作
    from config_nn import NN_INPUT_C as nn_input_c_direct
    assert nn_input_c_direct == NN_INPUT_C, \
        f'config_nn.NN_INPUT_C={nn_input_c_direct} != config.NN_INPUT_C={NN_INPUT_C}'
    print(f'  config_nn.NN_INPUT_C = {nn_input_c_direct} (与config一致)')

    # 3. 从各子模块能独立导入
    from config_game import TOTAL_TURN as tt, MAX_SCORE as ms
    from config_nn import NN_OUTPUT_C as no, MODEL_SIZE_LIMIT as msl
    from config_mcts import SEARCH_SINGLE_MAX as ssm, MCTS_POLICY_TEMPERATURE as mpt
    from config_train import LEARNING_RATE as lr, BATCH_SIZE as bs
    assert tt == 78 and ms == 60000
    assert no == 56 and msl == 376 * 1024
    assert ssm == 256 and mpt == 1.0
    assert lr == 7e-4 and bs == 1024
    print(f'  各子模块独立导入: config_game(TOTAL_TURN={tt}), '
          f'config_nn(NN_OUTPUT_C={no}), config_mcts(SEARCH_SINGLE_MAX={ssm}), '
          f'config_train(LEARNING_RATE={lr})')

    # 4. 验证config_nn独立导入NN维度常量
    from config_nn import NN_INPUT_C_GLOBAL, NN_INPUT_C_BC, NN_INPUT_C_RAMEN
    assert NN_INPUT_C_GLOBAL == 156, f'NN_INPUT_C_GLOBAL应为156, 实际{NN_INPUT_C_GLOBAL}'
    assert NN_INPUT_C_BC == 8, f'NN_INPUT_C_BC应为8, 实际{NN_INPUT_C_BC}'
    assert NN_INPUT_C_RAMEN == 16, f'NN_INPUT_C_RAMEN应为16, 实际{NN_INPUT_C_RAMEN}'
    print(f'  config_nn维度明细: GLOBAL={NN_INPUT_C_GLOBAL}, BC={NN_INPUT_C_BC}, RAMEN={NN_INPUT_C_RAMEN}')

    print('  [PASS] config拆分验证测试通过!')

if __name__ == "__main__":
    test_game()
    test_mcts()
    test_formula_layer()
    test_bad_condition()
    test_ramen_scenario()
    test_game_bc_integration()
    # 新增测试组
    test_mcts_full_game()
    test_handwritten_evaluator()
    test_nn_input_encoding()
    test_training_loss()
    # P2-1新增测试
    test_mcts_search_result_sanity()
    test_selfplay_value_labels()
    test_full_game_78turns()
    test_search_result_welford()
    test_config_split()
    print()
    print("="*50)
    print("全部验收通过! ✓")
