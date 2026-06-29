"""
赛马娘AI训练框架 - 事件处理模块

从game.py提取的事件处理函数。
包含随机事件、固定事件、友人卡事件等。
"""

from .action import GameStage
from .person import PersonType, FriendStage
from config import (
    TOTAL_TURN, EVENT_PROB,
    FRIEND_UNLOCK_PROB_LOW_JIBAN, FRIEND_UNLOCK_PROB_HIGH_JIBAN,
    FRIEND_TYPE_NONE, FRIEND_TYPE_YAYOI, FRIEND_TYPE_LIANGHUA,
    PSID_NONE, PSID_NONCARD_YAYOI, PSID_NONCARD_REPORTER, PSID_NPC,
    # 新增命名常量
    RANDOM_EVENT_PROB, MOTIVATION_DOWN_PROB, CHARA_EVENT_PROB,
    VITAL_EVENT_SMALL_PROB, VITAL_EVENT_BIG_PROB, MOTIVATION_UP_EVENT_PROB,
    URA_SKILL_SCORE, URA_START_TURN,
)


def check_random_events(game, rng):
    """模拟随机事件（含バッドコンディション公式层）"""
    if game.turn >= URA_START_TURN:
        return

    # ===== バッドコンディション回合效果 =====
    bc_effects = game.formula.bc_on_turn_start(game.turn, rng)
    if bc_effects['vital_drain'] != 0:
        game.add_vital(bc_effects['vital_drain'])
    if bc_effects['motivation_drain'] != 0:
        game.add_motivation(bc_effects['motivation_drain'])

    # ===== バッドコンディション随机获取 =====
    game.bc_manager.try_acquire_random(game.turn, rng)

    # ===== やる気下降事件（2.5周年后5回合不重复） =====
    if game.turn >= 12 and rng.random() < MOTIVATION_DOWN_PROB:
        if game.bc_manager.can_motivation_decrease(game.turn):
            game.add_motivation(-1)
            game.bc_manager.record_motivation_decrease(game.turn)

    # ===== 友人解锁出行 =====
    if game.friend_type != FRIEND_TYPE_NONE:
        p = game.persons[game.friend_person_id]
        if game.friend_stage == FriendStage.BEFORE_UNLOCK_OUTGOING:
            unlock_prob = (
                FRIEND_UNLOCK_PROB_HIGH_JIBAN if p.friendship >= 60
                else FRIEND_UNLOCK_PROB_LOW_JIBAN
            )
            if rng.random() < unlock_prob:
                handle_friend_unlock(game, rng)

    # ===== 支援卡连续事件（支援卡连续事件触发概率） =====
    if rng.random() < RANDOM_EVENT_PROB:
        card = rng.randint(0, 5)
        game.add_ji_ban(card, 5)
        game.add_status(rng.randint(0, 4), game.event_strength)
        game.skill_pt += game.event_strength
        if rng.random() < 0.4 * (1.0 - game.turn / TOTAL_TURN):
            # やる気上升受片頭痛限制
            if not game.bc_manager.is_motivation_blocked():
                game.add_motivation(1)
        if rng.random() < 0.5:
            game.add_vital(10)
        elif rng.random() < 0.06:
            game.add_vital(-10)

    # ===== 马娘随机事件（马娘随机事件概率） =====
    if rng.random() < CHARA_EVENT_PROB:
        game.add_all_status(3)

    # ===== 体力事件（小体力事件概率+大体力事件概率） =====
    if rng.random() < VITAL_EVENT_SMALL_PROB:
        game.add_vital(5)
    if rng.random() < VITAL_EVENT_BIG_PROB:
        game.add_vital(30)

    # ===== 心情事件（やる気上升事件概率，受片頭痛限制） =====
    if rng.random() < MOTIVATION_UP_EVENT_PROB:
        if not game.bc_manager.is_motivation_blocked():
            game.add_motivation(1)


def check_fixed_events(game, rng):
    """每回合的固定事件"""
    if game.is_refresh_mind:
        game.add_vital(5)
        if rng.random() < 0.25:
            game.is_refresh_mind = False

    if game.is_racing:
        if game.turn < 72:
            game._run_race(3, 45)
            game.add_ji_ban(PSID_NONCARD_YAYOI if game.friend_type != FRIEND_TYPE_YAYOI else game.friend_person_id, 4, ignore_ai_jiao=True)
        elif game.turn == 73:
            game._run_race(10, 40)
        elif game.turn == 75:
            game._run_race(10, 60)
        elif game.turn == 77:
            game._run_race(10, 80)

    if game.turn == 23:  # 第一年年底
        vital_space = game.max_vital - game.vital
        if vital_space >= 20:
            game.add_vital(20)
        else:
            game.add_all_status(5)

    elif game.turn == 29:  # 第二年继承
        for i in range(5):
            game.add_status(i, game.zhong_ma_blue_count[i] * 6)
        factor = rng.random() * 2
        for i in range(5):
            game.add_status(i, int(factor * game.zhong_ma_extra_bonus[i]))
        game.skill_pt += int((0.5 + 0.5 * factor) * game.zhong_ma_extra_bonus[5])
        for i in range(5):
            game.five_status_limit[i] += game.zhong_ma_blue_count[i] * 2
            game.five_status_limit[i] += rng.randint(0, 7)

    elif game.turn == 47:  # 第二年年底
        vital_space = game.max_vital - game.vital
        if vital_space >= 30:
            game.add_vital(30)
        else:
            game.add_all_status(8)

    elif game.turn == 48:  # 抽奖
        rd = rng.randint(0, 99)
        if rd < 16:
            game.add_vital(30)
            game.add_all_status(10)
            game.add_motivation(2)
        elif rd < 43:
            game.add_vital(20)
            game.add_all_status(5)
            game.add_motivation(1)
        elif rd < 89:
            game.add_vital(20)
        else:
            game.add_motivation(-1)

    elif game.turn == 49:
        game.skill_score += URA_SKILL_SCORE

    elif game.turn == 53:  # 第三年继承
        for i in range(5):
            game.add_status(i, game.zhong_ma_blue_count[i] * 6)
        factor = rng.random() * 2
        for i in range(5):
            game.add_status(i, int(factor * game.zhong_ma_extra_bonus[i]))
        game.skill_pt += int((0.5 + 0.5 * factor) * game.zhong_ma_extra_bonus[5])
        for i in range(5):
            game.five_status_limit[i] += game.zhong_ma_blue_count[i] * 2
            game.five_status_limit[i] += rng.randint(0, 7)

    elif game.turn == 70:
        game.skill_score += URA_SKILL_SCORE

    elif game.turn == 77:  # URA3结算
        # 记者
        if game.friendship_noncard_reporter >= 80:
            game.add_all_status(5)
            game.skill_pt += 20
        elif game.friendship_noncard_reporter >= 60:
            game.add_all_status(3)
            game.skill_pt += 10
        elif game.friendship_noncard_reporter >= 40:
            game.skill_pt += 10
        else:
            game.skill_pt += 5

        # 全胜检查
        if all(h == 2 for h in game.mecha_win_history):
            game.skill_pt += 40
            game.add_all_status(45)
            game.skill_pt += 175
        else:
            game.add_all_status(40)

        game.add_all_status(5)
        game.skill_pt += 20


def check_event_after_train(game, rng):
    """训练后检查事件并推进回合"""
    # P1-4修复: overdrive关闭已移至random_distribute_cards开头的恢复逻辑，不在事件后立即关闭
    maybe_update_deyilv(game)
    check_fixed_events(game, rng)
    check_random_events(game, rng)

    # 剧本回合结束
    if game.scenario is not None:
        game.scenario.on_turn_end(game, rng)

    # 回合+1
    game.turn += 1
    if game.turn < TOTAL_TURN:
        game.is_racing = game.is_racing_turn[game.turn]
    game.game_stage = GameStage.BEFORE_TRAIN


def maybe_update_deyilv(game):
    """更新得意率"""
    deyilv_bonus = 15 * game.mecha_upgrade[0][0]
    lianghua_enable = (
        game.friend_type == FRIEND_TYPE_LIANGHUA and
        game.friend_is_ssr and
        game.persons[game.friend_person_id].friendship >= 60 if game.friend_person_id >= 0 else False
    )
    if deyilv_bonus != game.current_deyilv_bonus or lianghua_enable != game.current_lianghua_effect_enable:
        game.current_deyilv_bonus = deyilv_bonus
        game.current_lianghua_effect_enable = lianghua_enable
        for i in range(6):
            game.persons[i].set_extra_deyilv_bonus(deyilv_bonus, lianghua_enable)


def handle_friend_click_event(game, rng, at_train: int):
    """友人点击事件"""
    if game.friend_type == FRIEND_TYPE_NONE:
        return

    if game.friend_stage == FriendStage.NOT_CLICKED:
        game.friend_stage = FriendStage.BEFORE_UNLOCK_OUTGOING
        if game.friend_type == FRIEND_TYPE_YAYOI:
            game.add_status_friend(0, 8)
            game.add_ji_ban(game.friend_person_id, 10)
            game.add_motivation(1)
        elif game.friend_type == FRIEND_TYPE_LIANGHUA:
            game.add_vital_max(4)
            game.add_ji_ban(game.friend_person_id, 10)
            game.add_motivation(1)


def _yayoi_outing_5(game, rng):
    """P1-10修复: Yayoi第5次外出，用单次随机判定保证结果一致性"""
    is_great = rng.random() < 0.75
    if is_great:
        game.add_vital_friend(30)
        game.add_status_friend(3, 36)
        game.skill_pt += 72
        game.is_refresh_mind = True
    else:
        game.add_vital_friend(26)
        game.add_status_friend(3, 24)
        game.skill_pt += 40
    game.add_motivation(1)
    game.add_ji_ban(game.friend_person_id, 5)


def handle_friend_outgoing(game, rng):
    """友人外出"""
    if game.friend_type == FRIEND_TYPE_YAYOI:
        outings = [
            # (体力回复, 干劲, 属性加成, pt加成)
            lambda: (game.add_vital_friend(30), game.add_motivation(1), game.add_status_friend(3, 20), game.add_ji_ban(game.friend_person_id, 5)),
            lambda: (game.add_vital_friend(30), game.add_motivation(1), game.add_status_friend(0, 10), game.add_status_friend(3, 10), setattr(game, 'is_refresh_mind', True), game.add_ji_ban(game.friend_person_id, 5)),
            lambda: (game.add_vital_friend(43) if game.max_vital - game.vital >= 20 else game.add_status_friend(3, 29), game.add_motivation(1), game.add_ji_ban(game.friend_person_id, 5)),
            lambda: (game.add_vital_friend(30), game.add_motivation(1), game.add_status_friend(3, 25), game.add_ji_ban(game.friend_person_id, 5)),
            # P1-10修复: 第5次外出用单次随机判定，避免体力/属性/pt结果不一致
            lambda: _yayoi_outing_5(game, rng),
        ]
        if game.friend_outgoing_used < len(outings):
            outings[game.friend_outgoing_used]()

    elif game.friend_type == FRIEND_TYPE_LIANGHUA:
        if game.friend_outgoing_used == 0:
            game.add_vital_friend(35)
            game.add_motivation(1)
            game.add_status_friend(0, 15)
            game.add_ji_ban(game.friend_person_id, 5)
        elif game.friend_outgoing_used == 1:
            game.add_vital_friend(30)
            game.add_motivation(1)
            game.add_status_friend(0, 10)
            game.add_status_friend(4, 10)
            game.add_ji_ban(game.friend_person_id, 5)
        elif game.friend_outgoing_used == 2:
            game.add_vital_friend(50)
            game.add_motivation(1)
            game.add_ji_ban(game.friend_person_id, 5)
        elif game.friend_outgoing_used == 3:
            game.add_vital_friend(30)
            game.add_motivation(1)
            game.add_status_friend(0, 25)
            game.add_ji_ban(game.friend_person_id, 5)
        elif game.friend_outgoing_used == 4:
            if rng.random() < 0.75:
                game.add_vital_friend(40)
                game.add_status_friend(0, 30)
                game.skill_pt += 72
            else:
                game.add_vital_friend(35)
                game.add_status_friend(0, 15)
                game.skill_pt += 40

    game.friend_outgoing_used += 1


def handle_friend_unlock(game, rng):
    """友人外出解锁"""
    if game.friend_type == FRIEND_TYPE_YAYOI:
        if game.max_vital - game.vital >= 15:
            game.add_vital_friend(25)
        else:
            game.add_status_friend(0, 8)
            game.add_status_friend(3, 8)
            game.skill_pt += 10
        game.add_motivation(1)
        game.is_refresh_mind = True
        game.add_ji_ban(game.friend_person_id, 5)
    elif game.friend_type == FRIEND_TYPE_LIANGHUA:
        game.add_vital_max(4)
        game.add_vital_friend(20)
        game.add_motivation(1)
        game.add_ji_ban(game.friend_person_id, 5)

    game.friend_stage = FriendStage.AFTER_UNLOCK_OUTGOING
