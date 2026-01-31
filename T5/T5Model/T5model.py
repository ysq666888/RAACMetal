from transformers import TFT5EncoderModel, T5Config
import tensorflow.keras.layers as layers
import tensorflow as tf
from pathlib import Path
import gc
root_path = str(Path(__file__).parent)


class T5Model(tf.keras.Model):
    def __init__(self, initial_range, embedding_dim) -> None:
        super().__init__(name='T5Model')
        self.initial_range = initial_range
        self.embedding_dim = embedding_dim
        model_config = T5Config.from_pretrained(f'{root_path}/ModelbaseConfig') 
        self.model = TFT5EncoderModel(model_config)

        self.classifier = GetLogits(initial_range=initial_range, embedding_dim=embedding_dim)
        self._input_signature = (
            tf.TensorSpec(shape=(None, 27), dtype=tf.int32, name='input_ids'),
            tf.TensorSpec(shape=(None, 27), dtype=tf.int32, name='attention_mask')
        )


    def train_step(self, data):
        inputs, targets = data
        with tf.GradientTape() as tape:
            logits: tf.Tensor = self(inputs, training=True)
            loss: tf.Tensor = self.compiled_loss(targets, logits, regularization_losses=self.losses)
        gc.collect()
        trainable_variables = self.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        self.compiled_metrics.update_state(targets, logits)
        gc.collect()
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        inputs, targets = data
        print(f'inputs: {type(inputs)}, {inputs[0].shape}')
        logits: tf.Tensor = self(inputs, training=False)
        self.compiled_metrics.update_state(targets, logits)
        return {m.name: m.result() for m in self.metrics}


    def predict_step(self, data):
        inputs, protein_data = data
        logits: tf.Tensor = self(inputs, training=False)
        return logits, protein_data

    @tf.autograph.experimental.do_not_convert
    def call(self, features, training=None, mask=None) -> tf.Tensor:
        if isinstance(features, (list, tuple)) and len(features) >= 2:
            input_ids = features[0]
            input_pad = features[1]
        elif isinstance(features, tf.Tensor):
            # features shape: (batch_size, 2, 27)
            input_ids = features[:, 0, :] 
            input_pad = features[:, 1, :]
        elif isinstance(features, dict):
            input_ids = features['input_ids']
            input_pad = features['input_pad']
        else:
            raise ValueError(f"Unexpected input format: {type(features)}")

        noise = tf.random.normal(shape=tf.shape(input_ids), stddev=1.0)
        input_ids = tf.cast(input_ids, tf.float32) + 0.05 * noise

        output: tf.Tensor =self.model(input_ids=input_ids, attention_mask=input_pad, training=training)
        last_hidden_state: tf.Tensor = output.last_hidden_state
        embedding_output = self.model.shared(input_ids)
        logits: tf.Tensor = self.classifier((last_hidden_state, embedding_output, input_pad), training=training)

        return logits

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            'initial_range':self.initial_range,
            'embedding_dim':self.embedding_dim
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    @property
    def input_signature(self):
        return self._input_signature
    
    def build_model(self):
        dummy_input = (
            tf.zeros((1, 27), dtype=tf.int32),
            tf.zeros((1, 27), dtype=tf.int32)
        )
        _ = self(dummy_input, training=False)
        print("Model built successfully with input signature:", self._input_signature)


class GetLogits(layers.Layer):
    def __init__(self, initial_range, embedding_dim) -> None:
        super().__init__(name='Get_loss')
        self.hidden_layer = layers.Dense(
            embedding_dim,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=initial_range),
            use_bias=False,
            activation=tf.keras.activations.swish,
            name='dis_output',
        )
        self.dropout = layers.Dropout(0.1)
        self.norm = tf.keras.layers.LayerNormalization()
        self.output_dense = layers.Dense(
            1,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=initial_range),
            use_bias=False, name='output_dense',
            activation=tf.keras.activations.swish,
            kernel_regularizer=tf.keras.regularizers.l1_l2(0.02),
        )
        
        

    def call(self, inputs_tuple: tuple, training=True) -> tf.Tensor:
        hidden_states: tf.Tensor = inputs_tuple[0]
        embedding_table: tf.Tensor = inputs_tuple[1]

        hidden: tf.Tensor = self.dropout(self.hidden_layer(hidden_states), training=training)
        hidden: tf.Tensor = tf.matmul(hidden, embedding_table, transpose_b=True)
        hidden = self.norm(hidden)
        hidden = tf.reduce_mean(hidden, axis=-1)
        logits: tf.Tensor = self.output_dense(hidden)
    
        return tf.squeeze(logits, -1)

