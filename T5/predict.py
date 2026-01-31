import tensorflow as tf
import os, argparse, gc
from data_make.utils import select_model
import pandas as pd
import numpy as np
from typing import Union
from T5Model.Configs import Config
from T5Model.T5model import T5Model, GetLogits
from start_and_output.Input_module import get_features
from start_and_output.output_compute import ComputeMetrics, compute_loss
tf.data.experimental.enable_debug_mode()
tf.config.optimizer.set_jit(True)

gpus = tf.config.list_physical_devices('GPU')
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
print("All GPUs:", gpus)

if gpus:
    try:
        tf.config.set_visible_devices(gpus[1], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[1], True)
    except RuntimeError as e:
        print(e)

tf.config.run_functions_eagerly(True)
random_seed = 42
tf.random.set_seed(random_seed)  # set random seed for tensorflow-cpu
os.environ['TF_DETERMINISTIC_OPS'] = '1'  # set random seed for tensorflow-gpu
print(tf.__version__)


def save_output(protein_data, score, output_path:str):
    pred = np.array(score > 0.5)
    df = pd.DataFrame({
    'protein_name': protein_data['protein_name'].astype(str),
    'center_res': protein_data['center_res'].astype(str),
    'score': score,
    'pred': pred
    })
    df.to_csv(output_path, index=False)


def predict(input_files: Union[dict, bool] = None, target_name='ca') -> None:
    config: Config = Config(train='predict', input_files=input_files, target_name=target_name)
    model: T5Model = T5Model(embedding_dim=config.embedding_dim, initial_range=config.initial_range)

    print(f"{'=' * 40}\n{'predict':^40}\n{'=' * 40}")
    data_eval = get_features(config, return_tensor_format=True)
    gc.collect()
    model = tf.keras.models.load_model(
        config.save_weights_path,
        custom_objects={
            'ComputeMetrics': ComputeMetrics, 
            'T5Model': T5Model, 
            'GetLogits': GetLogits,
            'compute_loss': compute_loss
        }
    )
    score, protein_data = model.predict(data_eval)
    save_output(protein_data, score=score, output_path=input_files['metrics_save_path'])
    
    print('model is predicted')


def main(input_file:str, save_dir: str, model_path):
    os.makedirs(save_dir, exist_ok=True)
    predict_dict:dict = {
        'input_file': input_file,
        'metrics_save_path': os.path.join(save_dir, 'predict.csv'),
        'save_weights_path': model_path
    }
    gc.collect()
    predict(input_files=predict_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predicting')
    parser.add_argument('-f', '--tfrecord_file', help='tfrecord_file path', required=True)
    parser.add_argument('-t', '--target_ion', help=f'ion ligand predicted', required=True)
    parser.add_argument('-s', '--save_dir', help='output save path', required=True)
    args = parser.parse_args()
    tfrecord_file = args.tfrecord_file
    target = args.target_ion
    save_dir = args.save_dir
    model_path = select_model(target_name=target)
    main(input_file=tfrecord_file, save_dir=save_dir, model_path=model_path)