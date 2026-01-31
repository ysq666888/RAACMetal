import tensorflow as tf
import os, time, gc
from typing import Union
from T5Model.Configs import Config
from T5Model.T5model import T5Model, GetLogits
from start_and_output.Input_module import get_features
from start_and_output.output_compute import ComputeMetrics, compute_loss
from T5Model.optimization import create_optimizer
tf.data.experimental.enable_debug_mode()
tf.config.optimizer.set_jit(True)
os.environ.pop("TF_USE_LEGACY_KERAS", None)

root_path = os.getcwd()
print(f'root path :{root_path}')

gpus = tf.config.list_physical_devices('GPU')
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
print("All GPUs:", gpus)

if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

tf.config.run_functions_eagerly(True)
random_seed = 42
tf.random.set_seed(random_seed)  # set random seed for tensorflow-cpu
os.environ['TF_DETERMINISTIC_OPS'] = '1'  # set random seed for tensorflow-gpu
print(tf.__version__)


def bug_logs(context: str) -> None:
    with open('bug_log.txt', 'a') as log:
        log.write(context)


def main(input_files: Union[dict, bool] = None, train: bool = True, target_name='ca') -> None:
    config: Config = Config(train=train, input_files=input_files, target_name=target_name)
    model: T5Model = T5Model(embedding_dim=config.embedding_dim, initial_range=config.initial_range)
    optimizer = create_optimizer(config)

    if train:

        print(f"{'=' * 40}\n{'Train':^40}\n{'=' * 40}")
        model.compile(
            optimizer=optimizer,
            loss=compute_loss,
            metrics=ComputeMetrics(
                save_path=config.metrics_save_path,
                return_metrics=config.return_metrics,
                threshold=config.thresholds),
        )
        train_ds, val_ds, dataset_steps = get_features(config, return_tensor_format=True)
        print(f'dataset_steps: {dataset_steps}')
        model.build_model()
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=config.train_epochs,
            steps_per_epoch=dataset_steps,
        )
        model.summary()
        print(f'{model.summary()}:summary')
        model.save(config.save_weights_path, save_format='tf')
        print('model saved')
    else:
        print(f"{'=' * 40}\n{'Test':^40}\n{'=' * 40}")
        data_eval, step = get_features(config, return_tensor_format=True)
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
        gc.collect()
        model.evaluate(data_eval, steps=1)
        gc.collect()
        print('model is evaluated')


def reduce(scale) -> None:
    reduce_dir: str = f'{root_path}/data_make/reduce'
    train_base_dir: str = os.path.join(reduce_dir, 'train')
    test_base_dir: str = train_base_dir.replace('train', 'test')

    targets: list = os.listdir(train_base_dir)
    current_target = {
        # 'fe2_train': 'type_51+size_19+LI-V-G-A-P-Q-N-M-T-S-C-E-D-K-R-Y-F-W-H', # plot_ok
        'co_train': 'type_63+size_8+QSNTGAP-ED-LIVM-FWY-H-R-K-C', # better, plot_o,
        'mg_train': 'type_11+size_10+VILMF-WY-A-C-ED-RK-G-P-ST-HQN', # plot_ok
        'ni_train': 'type_29+size_14+AST-C-DE-FL-G-H-IV-KR-M-N-P-Q-W-Y',
        # 'fe_train': 'type_7+size_11+W-C-G-P-H-NDE-RQK-AST-F-Y-VMIL', # better
        # 'cu_train': 'type_4+size_8+G-D-N-AEFILMKQRVWY-CH-T-S-P', # better plot_ok
        'mn_train': 'type_1+size_15+LVIM-C-A-G-S-T-P-FY-W-E-D-N-Q-KR-H', # plot_ok
        # 'k_train': 'type_49+size_17+C-FY-W-ML-IV-G-P-A-T-S-N-H-Q-E-D-R-K', # plot_ok
        # 'zn_train': 'type_63+size_8+QSNTGAP-ED-LIVM-FWY-H-R-K-C', # plot_ok
        'na_train': 'type_38+size_9+FWYH-ML-IV-CA-NTS-P-G-DE-QRK', # better plot_ok
        'ca_train': 'type_33+size_11+ST-A-ND-G-RQ-EK-H-P-IVLM-WYF-C' # plot_ok
    }
    print(current_target)
    for target in targets:
        if target not in current_target.keys():
            continue

        all_types: list = os.listdir(os.path.join(train_base_dir, target))
        target_name: str = target.split('_')[0]  # 获取前缀

        for reduce_type in all_types:
            # if reduce_type != current_target[target]:
            if reduce_type != 'type_0+size_20+A-C-D-E-F-G-H-I-K-L-M-N-P-Q-R-S-T-V-W-Y':
                continue
            print(f"Processing reduce_type: {reduce_type}")
            result_dir = f"{root_path}/results_{scale}/{target_name}/{reduce_type}" if scale else f"{root_path}/results/{target_name}/{reduce_type}"
            os.makedirs(result_dir, exist_ok=True)
            metrics_path: str = os.path.join(result_dir, 'metrics.txt') 
            save_weights_path: str = f"{root_path}/Models_{scale}/{target_name}/{reduce_type}" if scale else f"{root_path}/Models/{target_name}/{reduce_type}"
            # save_weights_path = f"{root_path}/noreduce_models/noreduce_{target_name}/"
            if os.path.exists(save_weights_path) and os.listdir(save_weights_path):
                # ---- Test ----
                with open(metrics_path, 'w') as f:
                    f.write('')
                test_file = f'test_{scale}.tfrecord' if scale else 'test.tfrecord'
                test_path: str = os.path.join(test_base_dir, f"{target.replace('train', 'test')}", reduce_type,test_file)

                test_dict: dict = {
                    'input_file': test_path,
                    'metrics_save_path': metrics_path,
                    'save_weights_path': save_weights_path,
                }
                main(input_files=test_dict, train=False)
            else:
                os.makedirs(save_weights_path, exist_ok=True)
                # ---- Train ----
                train_file = f'train_{scale}.tfrecord' if scale else 'train.tfrecord'
                train_path: str = os.path.join(train_base_dir, target, reduce_type, train_file)
                print(f'Training on: {train_path}')

                train_dict: dict = {
                    'input_file': train_path,
                    'metrics_save_path': metrics_path,
                    'save_weights_path': save_weights_path,
                }
                main(input_files=train_dict, train=True)
                with open(metrics_path, 'w') as f:
                    f.write('')
                # ---- Test ----
                with open(metrics_path, 'w') as f:
                    f.write('')
                test_file = f'test_{scale}.tfrecord' if scale else 'test.tfrecord'
                test_path: str = os.path.join(test_base_dir, f"{target.replace('train', 'test')}", reduce_type, test_file)

                test_dict: dict = {
                    'input_file': test_path,
                    'metrics_save_path': metrics_path,
                    'save_weights_path': save_weights_path,
                }
                main(input_files=test_dict, train=False)

                


if __name__ == '__main__':
    start = time.time()
    run_reduce = True
    scale = 5

    try:
        if run_reduce:
            reduce(scale)
        else:
            targets = ['cu']
            for target in targets:
                print(f'target: {target}')
                main(train=True, target_name = target)
                main(train=False, target_name = target)
    except Exception as e:
        print(f'error :{e}')
    finally:
        end = time.time()
        print(f'time: {end - start}')


