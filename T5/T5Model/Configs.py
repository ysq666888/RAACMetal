import os
from typing import Union
		
class Config:
	def __init__(
			self,
			train: bool = True,
			input_files: Union[dict, bool] = None,
			target_name: str = 'ca',) -> None:
		
		# 2): run module
		self.train: bool = train
		self.train_epochs: int = 20
		
		"""Input module"""
		if input_files is None:
			os.makedirs(f'./results', exist_ok=True)
			self.metrics_save_path: str = f'./results_5/noreduce_metrics_{target_name}.txt'
			self.save_weights_path: str = f'./Models/noreduce_{target_name}'
			os.makedirs(self.save_weights_path, exist_ok=True)
			if train:
				self.input_file: str = f'./data_make/noreduce/train/{target_name}_train/train.tfrecord'
			else:
				self.input_file: str = f'./data_make/noreduce/test/{target_name}_test/test.tfrecord'
				with open(self.metrics_save_path, 'w') as f:
					f.write('')
		else:
			self.input_file: str = input_files.get('input_file')
			
			self.metrics_save_path: str = input_files.get('metrics_save_path')
			print(f'metrics:{self.metrics_save_path}')
			self.save_weights_path: str = input_files.get('save_weights_path')
		# cu: 100, fe: 100, na: 200, fe2: 200, co: 20, k: 500, mg: 500, zn: 500, mn: 500
		self.batch_size: int = 100
		self.buffer_size: int = 100000000
		self.max_seq_length: int = 27

		self.embedding_dim = 1024
		
		
		"""compute loss"""
		self.return_metrics: str = 'auc_roc'
		self.staircase: bool = True
		self.thresholds: float = 0.5
		self.initial_learning_rate: float = 0.05
		self.learning_rate_decay_factor: float = 0.95
		self.learning_rate_decay_steps: int = 30
		self.initial_range = 0.2
		