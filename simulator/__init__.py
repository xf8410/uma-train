"""赛马娘AI训练框架 - 模拟器包"""

from simulator.game import Game
from simulator.action import Action, TrainActionType, GameStage
from simulator.person import Person, PersonType, SupportCard, CardTrainingEffect, FriendStage
from simulator.bad_condition import BadConditionType, BadCondition, BadConditionManager
from simulator.formula import FormulaLayer, MOTIVATION_MULTIPLIER, MOTIVATION_REST_VITAL
from simulator.scenarios.base import ScenarioBase
from simulator.scenarios.dreams import DreamsScenario
