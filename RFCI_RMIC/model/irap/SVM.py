# import packages
import os, csv, multiprocessing, sys, joblib
file_path = os.path.dirname(__file__)
sys.path.append(file_path)
import numpy as np
from sklearn import svm
from joblib import dump, load
import Load as iload
from tqdm import tqdm  
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    matthews_corrcoef, f1_score, recall_score,
    roc_curve, precision_recall_curve
)
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import KFold



def plot_curves(y_true, y_score, model_name, curve_path):
    
    base_filename = model_name.lower()
    
    # 保存格式配置
    formats = [
        ('png', {'dpi': 300}),                    
        ('pdf', {'format': 'pdf', 'dpi': 300}),    
        ('svg', {'format': 'svg', 'dpi': 300}),    
        ('tiff', {'format': 'tiff', 'dpi': 300})   #
    ]
    
    # ROC曲线
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_roc = roc_auc_score(y_true, y_score)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_roc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} ROC Curve")
    plt.legend()
    plt.tight_layout()
    
    # 保存
    for fmt, save_kwargs in formats:
        plt.savefig(
            os.path.join(curve_path, f'{base_filename}_roc.{fmt}'),
            **save_kwargs
        )
    
    plt.close()
    
    # PR曲线
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    auc_pr = average_precision_score(y_true, y_score)
    
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"AUC-PR = {auc_pr:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{model_name} PR Curve")
    plt.legend()
    plt.tight_layout()
    
    # 保存
    for fmt, save_kwargs in formats:
        plt.savefig(
            os.path.join(curve_path, f'{base_filename}_pr.{fmt}'),
            **save_kwargs
        )
    
    plt.close()
    
    saved_files = []
    for fmt, _ in formats:
        saved_files.append(f'{base_filename}_roc.{fmt}')
        saved_files.append(f'{base_filename}_pr.{fmt}')
    
    return saved_files


def svm_grid(train_data, train_label, test_data, test_label, model_path=None):
    # 如果有模型保存路径，检查模型是否存在
    if model_path and os.path.exists(model_path):
        print(f"加载已保存的SVM模型: {model_path}")
        model = joblib.load(model_path)
    else:
        print("训练新的SVM模型...")
        my_svm = svm.SVC(C=1.0, gamma='scale', decision_function_shape="ovo", 
                        random_state=0, probability=True)
        model = my_svm.fit(train_data, train_label)
        
        # 保存模型
        if model_path:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump(model, model_path)
            print(f"SVM模型已保存到: {model_path}")
    
    y_pred = model.predict(test_data)
    y_score = model.predict_proba(test_data)[:, 1]
    
    return {
        'accuracy': accuracy_score(test_label, y_pred),
        'auc_roc': roc_auc_score(test_label, y_score),
        'auc_pr': average_precision_score(test_label, y_score),
        'mcc': matthews_corrcoef(test_label, y_pred),
        'f1': f1_score(test_label, y_pred),
        'recall': recall_score(test_label, y_pred)
    }, test_label, y_score


def mlp_grid(train_data, train_label, test_data, test_label, model_path=None):
    if model_path and os.path.exists(model_path):
        print("训练新的MLP模型...")
        my_mlp = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation="relu",
            alpha=0.0001,
            solver="adam",
            learning_rate_init=0.001,
            max_iter=500,
            random_state=0
        )
        model = my_mlp.fit(train_data, train_label)
        
        if model_path:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump(model, model_path)
            print(f"MLP模型已保存到: {model_path}")
    else:
        print("训练新的MLP模型...")
        my_mlp = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation="relu",
            alpha=0.0001,
            solver="adam",
            learning_rate_init=0.001,
            max_iter=500,
            random_state=0
        )
        model = my_mlp.fit(train_data, train_label)
        
        if model_path:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump(model, model_path)
            print(f"MLP模型已保存到: {model_path}")
    
    y_pred = model.predict(test_data)
    y_score = model.predict_proba(test_data)[:, 1]
    
    return {
        'accuracy': accuracy_score(test_label, y_pred),
        'auc_roc': roc_auc_score(test_label, y_score),
        'auc_pr': average_precision_score(test_label, y_score),
        'mcc': matthews_corrcoef(test_label, y_pred),
        'f1': f1_score(test_label, y_pred),
        'recall': recall_score(test_label, y_pred)
    }, test_label, y_score


def xgb_grid(train_data, train_label, test_data, test_label, model_path=None):
    if model_path and os.path.exists(model_path):
        print("训练新的XGBoost模型...")
        my_xgb = XGBClassifier(
            n_estimators=100,       
            max_depth=6,          
            learning_rate=0.1,   
            subsample=0.8,         
            colsample_bytree=0.8,   
            objective="binary:logistic",  
            random_state=0,
            n_jobs=-1
        )
        model = my_xgb.fit(train_data, train_label)
        
        if model_path:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump(model, model_path)
            print(f"XGBoost模型已保存到: {model_path}")
    else:
        print("训练新的XGBoost模型...")
        my_xgb = XGBClassifier(
            n_estimators=100,       
            max_depth=6,          
            learning_rate=0.1,   
            subsample=0.8,         
            colsample_bytree=0.8,   
            objective="binary:logistic",  
            random_state=0,
            n_jobs=-1
        )
        model = my_xgb.fit(train_data, train_label)
        
        if model_path:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump(model, model_path)
            print(f"XGBoost模型已保存到: {model_path}")
    
    y_pred = model.predict(test_data)
    y_score = model.predict_proba(test_data)[:, 1]
    
    return {
        'accuracy': accuracy_score(test_label, y_pred),
        'auc_roc': roc_auc_score(test_label, y_score),
        'auc_pr': average_precision_score(test_label, y_score),
        'mcc': matthews_corrcoef(test_label, y_pred),
        'f1': f1_score(test_label, y_pred),
        'recall': recall_score(test_label, y_pred)
    }, test_label, y_score

rmic_xgb: dict = {
    'Zn': 't20s12',
    'Co': 't12s12',
    'Cu': 't39s10',
    'Ca': 't53s14',
    'K': 't46s5',
    'Mg': 't66s11',
    'Mn': 't32s10',
    'Feijinshulizi': 't29s14',
    'Fe': 't6s14',
    'Ni': 't4s14',
    'Na': 't41s10'
}

rfci_dict: dict = {
    'Ca': 't32s9',
    'Co': 't51s14',
    'Cu': 't40s8',
    'Fei': 't33s12',
    'K': 't6s14',
    'Mg': 't33s11',
    'Mn': 't5s11',
    'Na': 't34s11',
    'Ni': 't9s15',
    'Zn': 't31s16'
}

rmic_mlp = {
    'Cu': 't63s11',
    'Fe': 't32s10',
    'K': 't1s12',
    'Mg': 't4s11',
    'Na': 't63s9',
    'Ni': 't1s10',
    'Zn': 't35s16',
    'Co': 't58s18',
    'Feijinshulizi': 't32s17',
    'Mn': 't34s15',
    'Ca': 't2s7'
}

def balance_check_data(data, label, scale):
    rng = np.random.default_rng(42)

    pos_idx = np.where(label == 1)[0]
    neg_idx = np.where(label == 0)[0]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    print(f'n_pos:{pos_idx.shape}, n_neg:{neg_idx.shape}')
    if n_pos > n_neg:
        label = 1 - label
        pos_idx = np.where(label == 1)[0]
        neg_idx = np.where(label == 0)[0]
        n_pos = len(pos_idx)
        n_neg = len(neg_idx)
    if scale:
        print('will scale')
        target_neg = int(n_pos * ((10-scale)/scale))

        if n_neg <= target_neg:
            chosen_neg = neg_idx
        else:
            # 随机采样负样本
            chosen_neg = rng.choice(neg_idx, size=target_neg, replace=False)

        final_idx = np.concatenate([pos_idx, chosen_neg])
        rng.shuffle(final_idx)  

        return data[final_idx], label[final_idx]
    else:
        return data, label

def write_model_results_to_csv(writer, dataset_name, model_name, metrics_dict, five_fold_num=False):
    if five_fold_num:
        writer.writerow([
        five_fold_num,
        dataset_name,
        model_name,
        metrics_dict.get('auc_roc', ''),
        metrics_dict.get('auc_pr', ''),
        metrics_dict.get('mcc', ''),
        metrics_dict.get('f1', ''),
        metrics_dict.get('recall', ''),
        metrics_dict.get('accuracy', '')
        ])
    else:
        writer.writerow([
            dataset_name,
            model_name,
            metrics_dict.get('auc_roc', ''),
            metrics_dict.get('auc_pr', ''),
            metrics_dict.get('mcc', ''),
            metrics_dict.get('f1', ''),
            metrics_dict.get('recall', ''),
            metrics_dict.get('accuracy', '')
        ])


def process_single_file(file_info):
    i, out, model_funcs, model_select, scale, save_model, five_fold = file_info
    results_for_file = []
    try:
        
        ion = os.path.basename(os.path.dirname(i)).split('_')[-1]

        train_data, train_label = iload.load_svmfile(i)
        train_data, train_label = balance_check_data(train_data, train_label, scale=scale)
        root_path = os.path.dirname(os.path.dirname(i))

        if scale:
            curves_path = os.path.join(root_path, f'curves_scale_{scale}', f'{os.path.basename(i).split(".")[0]}', ion)
            save_model_dir = os.path.join(root_path, f'models_{scale}', f'{os.path.basename(i).split(".")[0]}', ion)
        else:
            curves_path = os.path.join(root_path, 'curves', f'{os.path.basename(i).split(".")[0]}', ion)
            save_model_dir = os.path.join(root_path, f'models', f'{os.path.basename(i).split(".")[0]}', ion)

        os.makedirs(curves_path, exist_ok=True)
        if save_model:
            os.makedirs(save_model_dir, exist_ok=True)

        if five_fold:
            X = np.array(train_data)
            y = np.array(train_label)
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            num = 0
            outputs = {**{f'{m}_score': [] for m in model_select}, **{f'{m}_label': [] for m in model_select}}

            for train_idx, val_idx in kf.split(X, y):
                train_data, test_data = X[train_idx], X[val_idx]
                train_label, test_label = y[train_idx], y[val_idx]
                for model in model_select:
                    model_path = os.path.join(save_model_dir, f'{model}_{num}.joblib')
                    output, temp_label, temp_score = model_funcs[model](train_data, train_label, test_data, test_label, model_path)
                    plot_curves(temp_label, temp_score, f'{model.upper()}_{num}', curves_path)
                    outputs[f'{model}_score'].append(temp_score)
                    outputs[f'{model}_label'].append(temp_label)
                    with open(out, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f, delimiter='\t')
                        write_model_results_to_csv(writer, i, f'{model.upper()}', output, str(num))
                    results_for_file.append({
                        'fold': num,
                        'dataset': i,
                        'model': model.upper(),
                        **output
                    })
                num += 1
            # 汇总五折结果
            for model in model_select:
                all_labels = np.concatenate([np.asarray(x) for x in outputs[f'{model}_label']])
                all_scores = np.concatenate([np.asarray(x) for x in outputs[f'{model}_score']])
                plot_curves(all_labels, all_scores, f'{model.upper()}_five_fold', curves_path)

        else:
            test_data, test_label = iload.load_svmfile(i.replace('Train', 'Test'))
            test_data, test_label = balance_check_data(test_data, test_label, scale=scale)
            for model in model_select:
                model_path = os.path.join(save_model_dir, f'{model}.joblib')
                output, temp_label, temp_score = model_funcs[model](train_data, train_label, test_data, test_label, model_path)
                plot_curves(temp_label, temp_score, f'{model.upper()}', curves_path)
                with open(out, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f, delimiter='\t')
                    write_model_results_to_csv(writer, i, f'{model.upper()}', output)
                results_for_file.append({
                    'fold': None,
                    'dataset': i,
                    'model': model.upper(),
                    **output
                })

        return results_for_file

    except Exception as e:
        return [{
            'fold': None,
            'dataset': i,
            'model': None,
            'status': 'error',
            'message': str(e)
        }]


def svm_grid_folder(path, model_type, out=False, model_select=['xgb'], scale=3, save_model=True, five_fold=True):
    model_funcs = {
        'svm': svm_grid,
        'mlp': mlp_grid,
        'xgb': xgb_grid
    }
    max_workers = multiprocessing.cpu_count() // 2

    # 初始化输出文件
    if out:
        if not out.endswith('.csv'):
            out = out.rsplit('.', 1)[0] + '.csv'
        if scale:
            out = out.replace('.csv', f'_scale_{scale}.csv')
        if five_fold:
            out = out.split('.')[0] + '_five_fold.csv'
            header = ['Five_fold', 'Dataset', 'Model', 'ROC', 'PR', 'MCC', 'F1', 'RECALL', 'ACCURACY']
        else:
            header = ['Dataset', 'Model', 'ROC', 'PR', 'MCC', 'F1', 'RECALL', 'ACCURACY']
        with open(out, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(header)

    file_infos = []
    for i in path:
        if not i.endswith('.rap'):
            continue
        ion = os.path.basename(os.path.dirname(i)).split('_')[-1]
        if 't0s20' not in i:
            pass
        if model_type == 'rmic':
            if ion in rmic_xgb.keys():
                if rmic_xgb[ion] not in i:
                    continue
        elif model_type == 'rfci':
            if ion in rfci_dict.keys():
                if rfci_dict[ion] not in i:
                    pass
        file_infos.append((i, out, model_funcs, model_select, scale, save_model, five_fold))

    all_results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(process_single_file, info): info[0] for info in file_infos}
        with tqdm(total=len(file_infos), desc="并行处理进度", unit="文件") as pbar:
            for future in as_completed(future_to_file):
                file = future_to_file[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                    pbar.set_postfix_str(f"完成: {os.path.basename(file)[:20]}...")
                except Exception as e:
                    pbar.set_postfix_str(f"错误: {os.path.basename(file)[:20]}...")
                    print(f"\n处理文件 {file} 时出错: {e}")
                pbar.update(1)

