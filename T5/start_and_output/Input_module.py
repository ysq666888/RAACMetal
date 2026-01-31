import tensorflow as tf
from typing import Union


def get_dataset_size(dataset):
    count = 0
    for _ in dataset:
        count += 1
    return count

def get_features(config, epochs: Union[bool, int] = False, return_tensor_format: bool = False) -> tf.data.Dataset:
	
	def _parse_example(input_example) -> tuple[dict[str, tf.Tensor], tf.Tensor]:
		if config.train != 'predict':
			examples = tf.io.parse_single_example(input_example, features=feature_description)
			labels: tf.Tensor = examples['labels']
			
			if return_tensor_format:
				return (examples['input_ids'], examples['input_mask']), labels
			else:
				
					features_dict = {
						'protein_name': examples['protein_name'],
						'input_ids': examples['input_ids'],
						'input_pad': examples['input_mask'] 
					}
					return features_dict, labels
		else:
			predict_description = {
				'protein_name': tf.io.FixedLenFeature([], tf.string),  
				'center_res': tf.io.FixedLenFeature([], tf.string),   
				'input_ids': tf.io.FixedLenFeature([config.max_seq_length], tf.int64), 
				'input_mask': tf.io.FixedLenFeature([config.max_seq_length], tf.int64), 
			}
			examples = tf.io.parse_single_example(input_example, features=predict_description)
			if return_tensor_format:
				return (examples['input_ids'], examples['input_mask']), {'protein_name':examples['protein_name'], 'center_res':examples['center_res']}
			else:
				
				features_dict = {
					'protein_name': examples['protein_name'],
					'input_ids': examples['input_ids'],
					'input_pad': examples['input_mask'],
					'center_res':examples['center_res']
				}
				return features_dict
	
	input_file: str = config.input_file
	print(f'inpput_file:{input_file}')
	
	if epochs:
		epochs: int = epochs
	else:
		epochs: int = config.train_epochs
	
	dataset: tf.data.TFRecordDataset = tf.data.TFRecordDataset([input_file])
	
	feature_description: dict = {
		'protein_name': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
		'input_ids': tf.io.FixedLenFeature([config.max_seq_length], tf.int64),
		'input_mask': tf.io.FixedLenFeature([config.max_seq_length], tf.int64),
		'labels': tf.io.FixedLenFeature([config.max_seq_length], tf.int64),
	}
	data_set: tf.data.Dataset = dataset.map(_parse_example)
	
	if config.train  and config.train != 'predict':
		train_ds, val_ds = tf.keras.utils.split_dataset(
			data_set,
			left_size=0.95,  
			right_size=0.05,  
			shuffle=True,  
			seed=42
		)
		train_size = train_ds.cardinality().numpy()
		train_steps = min(train_size // config.batch_size, 120)
		val_size = val_ds.cardinality().numpy()
		print(f'train_ds:{train_size} val_ds:{val_size}')
		train_ds: tf.data.Dataset = train_ds.shuffle(
			buffer_size=config.buffer_size).batch(config.batch_size, drop_remainder=True).repeat(epochs).prefetch(tf.data.AUTOTUNE)
		val_ds = val_ds.batch(val_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
		return train_ds, val_ds, train_steps
	else:
		test_size = get_dataset_size(data_set)
		dataset: tf.data.TFRecordDataset = tf.data.TFRecordDataset([input_file])
		data_set: tf.data.Dataset = dataset.map(_parse_example)
		print(f'test_ds:{test_size}')
		if test_size > 1e5:
			test_size = int(1e5)
			step = test_size // int(1e5)
		else:
			step = 1
		data_set: tf.data.Dataset = data_set.batch(test_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
		
		return data_set, step
	
