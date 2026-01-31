import tensorflow as tf
import tensorflow.keras.backend as K
import os
from start_and_output.plot import plot_auc_pr, plot_auc_roc

def compute_loss(y_true: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
    y_true = tf.cast(y_true, tf.float32)[:, 13]
    loss = tf.losses.MSE(y_true, logits)
    return loss

class ComputeMetrics(tf.keras.metrics.Metric):
    def __init__(self,
                 save_path: str,
                 return_metrics: str,
                 threshold: float = 0.5,**kwargs) -> None:
        super().__init__(**kwargs)
        self.save_path = save_path
        self.plot_path = os.path.dirname(self.save_path).replace('results', 'plot')
        os.makedirs(self.plot_path, exist_ok=True)
        print(f'plot_path:{self.plot_path}, save_path:{self.save_path}')
        self.return_metrics = return_metrics
        self.threshold = threshold
        self.auc_roc   = tf.keras.metrics.AUC(curve='ROC', name='auc_roc')
        self.auc_pr    = tf.keras.metrics.AUC(curve='PR',  name='auc_pr')
        self.tp        = tf.keras.metrics.TruePositives(
                            name='tp', thresholds=self.threshold)
        self.tn        = tf.keras.metrics.TrueNegatives(
                            name='tn', thresholds=self.threshold)
        self.fp        = tf.keras.metrics.FalsePositives(
                            name='fp', thresholds=self.threshold)
        self.fn        = tf.keras.metrics.FalseNegatives(
                            name='fn', thresholds=self.threshold)
        self.precision = tf.keras.metrics.Precision(
                            name='precision', thresholds=self.threshold)
        self.recall    = tf.keras.metrics.Recall(
                            name='recall', thresholds=self.threshold)
        self.accuracy  = tf.keras.metrics.BinaryAccuracy(
                            name='accuracy', threshold=self.threshold)
        self._order = [
          'auc_roc', 'auc_pr',
          'precision','recall','f1','mcc',
          'accuracy'
        ]
        self.results = {}
        self.history = {
            'y_prob': [],
            'y_true': [],
        }

    def update_state(self,
                     y_true: tf.Tensor,
                     y_prob: tf.Tensor,
                     sample_weight=None,
                     ) -> None:
        # y_prob = tf.sigmoid(y_prob)
        y_true  = tf.cast(y_true, tf.float32)[:, 13]

        self.history['y_prob'] += list(y_prob.numpy())
        self.history['y_true'] += list(y_true.numpy())
        plot_auc_roc(self.history['y_true'], self.history['y_prob'], self.plot_path)
        plot_auc_pr(self.history['y_true'], self.history['y_prob'], self.plot_path)
        self.auc_roc.update_state(y_true, y_prob)
        self.auc_pr .update_state(y_true, y_prob)

        self.tp.update_state(y_true, y_prob)
        self.tn.update_state(y_true, y_prob)
        self.fp.update_state(y_true, y_prob)
        self.fn.update_state(y_true, y_prob)

        self.precision.update_state(y_true, y_prob)
        self.recall   .update_state(y_true, y_prob)
        self.accuracy .update_state(y_true, y_prob)

        auc_r = self.auc_roc.result()
        auc_p = self.auc_pr .result()
        prec  = self.precision.result()
        rec   = self.recall.result()
        tp    = tf.cast(self.tp.result(), tf.float32)
        tn    = tf.cast(self.tn.result(), tf.float32)
        fp    = tf.cast(self.fp.result(), tf.float32)
        fn    = tf.cast(self.fn.result(), tf.float32)

        f1 = 2. * prec * rec / (prec + rec + K.epsilon())
        num = tp * tn - fp * fn
        den = tf.sqrt(
            (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        ) + K.epsilon()
        mcc = num / den

        self.results = {
          'auc_roc':   auc_r,
          'auc_pr':    auc_p,
          'precision': prec,
          'recall':    rec,
          'f1':        f1,
          'mcc':       mcc,
          'accuracy':  self.accuracy.result()
        }

        self._write_metrics()

    def _write_metrics(self):
        dirname = os.path.dirname(self.save_path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)

        line = '\t'.join(
            f'{self.results[k].numpy():.6f}'
            for k in self._order
        )
        print(f'\nline:{line}\n')
        mode = 'a' if os.path.exists(self.save_path) else 'w'
        with open(self.save_path, mode) as f:
            f.write(line + '\n')

    def result(self) -> tf.Tensor:
        return self.results[self.return_metrics]

    def reset_state(self):
        # reset every sub‐metric
        for m in [
            self.auc_roc, self.auc_pr,
            self.tp, self.tn, self.fp, self.fn,
            self.precision, self.recall, self.accuracy
        ]:
            m.reset_state()

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "save_path": self.save_path,
            "return_metrics": self.return_metrics,
            "threshold": self.threshold
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
