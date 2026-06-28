"""训练模块"""
from training.train import train, calculate_loss, cross_entropy_loss
from training.dataset import UmaTrainDataset, SelfPlayDataset, generate_random_data
from training.selfplay import SelfPlayWorker
