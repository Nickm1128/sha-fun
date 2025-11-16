import tensorflow as tf
import numpy as np

@tf.keras.utils.register_keras_serializable(package="custom")
class PerFeatureGaussianNoise(tf.keras.layers.Layer):
    def __init__(self, stdvec, frac_start=0.10, name="train_noise", **kwargs):
        super().__init__(name=name, **kwargs)
        stdvec = np.asarray(stdvec).reshape(1, -1).astype("float32")
        self.stdvec_init = tf.convert_to_tensor(stdvec, dtype=tf.float32)
        self.frac_start = float(frac_start)

    def build(self, input_shape):
        self.stdvec = self.add_weight(
            name="stdvec",
            shape=self.stdvec_init.shape,
            initializer=tf.keras.initializers.Constant(self.stdvec_init),
            trainable=False,
        )
        self.frac_var = self.add_weight(
            name="frac",
            shape=(),
            initializer=tf.keras.initializers.Constant(self.frac_start),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, x, training=False):
        if training:
            noise = tf.random.normal(tf.shape(x), dtype=x.dtype)
            return x + noise * (self.frac_var * self.stdvec)
        return x

@tf.keras.utils.register_keras_serializable(package="custom")
class SineDense(tf.keras.layers.Layer):
    def __init__(self, units, w0=1.0, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.w0 = float(w0)

    def build(self, input_shape):
        self.dense = tf.keras.layers.Dense(self.units, use_bias=True,
                                           kernel_initializer=tf.keras.initializers.RandomUniform(-1.0,1.0))
        super().build(input_shape)

    def call(self, x):
        return tf.sin(self.w0 * self.dense(x))

class SinePolicyMultiAsset(tf.keras.Model):
    def __init__(self, input_dim: int, M: int, K: int,
                 hidden=(128,128,256,256),
                 noise_stdvec=None,
                 noise_frac_start=0.15,
                 learnable_cash_bias: bool = False):
        super().__init__()
        self.M = int(M)
        self.K = int(K)
        layers = []
        if noise_stdvec is not None:
            layers.append(PerFeatureGaussianNoise(noise_stdvec, frac_start=noise_frac_start, name="train_noise"))
        for h in hidden:
            layers.append(SineDense(h, w0=1.0))
        self.trunk = tf.keras.Sequential(layers, name="trunk")
        # K horizon heads for coins (logits)
        self.heads = [tf.keras.layers.Dense(self.M, name=f"coin_logits_h{k}") for k in range(self.K)]
        # gate over horizons
        self.gate = tf.keras.layers.Dense(self.K, activation='softmax', name="gate")
        # optional cash bias
        if learnable_cash_bias:
            self.cash_bias = tf.Variable(0.0, name="cash_bias", trainable=True, dtype=tf.float32)
        else:
            self.cash_bias = None
        self._built_for = input_dim
        # Build by calling once
        dummy = tf.zeros((1, input_dim), dtype=tf.float32)
        _ = self.call(dummy, training=False)

    def call(self, x, training=False):
        z = self.trunk(x, training=training)
        # per-horizon allocations over coins
        per_h_allocs = []
        for k, head in enumerate(self.heads):
            logits = head(z)  # (B, M)
            per_h_allocs.append(tf.nn.softmax(logits, axis=-1))  # coin-only softmax for analysis
        per_h_allocs = tf.stack(per_h_allocs, axis=1)  # (B, K, M)
        gate_w = self.gate(z)  # (B, K)
        gate_w_exp = tf.expand_dims(gate_w, axis=-1)   # (B, K, 1)
        # blend coin logits via gate
        blended = tf.reduce_sum(per_h_allocs * gate_w_exp, axis=1)  # (B, M)
        # add cash and softmax over M+1
        if self.cash_bias is None:
            cash = tf.zeros((tf.shape(blended)[0], 1), dtype=blended.dtype)
        else:
            cash = tf.ones((tf.shape(blended)[0], 1), dtype=blended.dtype) * tf.sigmoid(self.cash_bias)
        logits_full = tf.concat([blended, cash], axis=-1)  # (B, M+1)
        alloc = tf.nn.softmax(logits_full, axis=-1)  # final allocation
        return alloc, [a for a in tf.unstack(per_h_allocs, axis=1)], gate_w

def build_policy_multi_asset(input_dim: int, M: int, K: int,
                             hidden=(128,128,256,256),
                             noise_stdvec=None, noise_frac_start=0.15,
                             learnable_cash_bias=False) -> SinePolicyMultiAsset:
    return SinePolicyMultiAsset(
        input_dim=input_dim, M=M, K=K, hidden=hidden,
        noise_stdvec=noise_stdvec, noise_frac_start=noise_frac_start,
        learnable_cash_bias=learnable_cash_bias
    )


# =============================
# State prediction models
# =============================

class SineStatePredictor(tf.keras.Model):
    """
    Model 1: Predict per-round SHA-256 working states (a..h) from final hash features.
    Input:  features of shape (D,) typically D=8 from hash_to_features
    Output: predictions of shape (64, 8)
    """
    def __init__(self, input_dim: int, hidden=(256, 256, 256), w0: float = 1.0, dropout: float = 0.0):
        super().__init__()
        layers = []
        for h in hidden:
            layers.append(SineDense(h, w0=w0))
            if dropout and dropout > 0:
                layers.append(tf.keras.layers.Dropout(dropout))
        self.trunk = tf.keras.Sequential(layers, name="state_trunk")
        self.out = tf.keras.layers.Dense(64 * 8, name="state_out")
        # build by calling once
        _ = self.call(tf.zeros((1, input_dim), dtype=tf.float32), training=False)

    def call(self, x, training=False):
        z = self.trunk(x, training=training)
        y = self.out(z)
        y = tf.reshape(y, (-1, 64, 8))
        return y


def build_sine_state_predictor(input_dim: int,
                               hidden=(256, 256, 256),
                               w0: float = 1.0,
                               dropout: float = 0.0) -> SineStatePredictor:
    return SineStatePredictor(input_dim=input_dim, hidden=hidden, w0=w0, dropout=dropout)


class SineStateRefiner(tf.keras.Model):
    """
    Model 2: Refine predicted states using final hash and initial predictions.
    Input:  concat([hash_features (D)], [pred_states_flat (64*8)])
    Output: refined states shape (64, 8)
    """
    def __init__(self, input_dim: int, hidden=(256, 256), w0: float = 1.0, dropout: float = 0.0):
        super().__init__()
        layers = []
        for h in hidden:
            layers.append(SineDense(h, w0=w0))
            if dropout and dropout > 0:
                layers.append(tf.keras.layers.Dropout(dropout))
        self.trunk = tf.keras.Sequential(layers, name="refine_trunk")
        self.out = tf.keras.layers.Dense(64 * 8, name="refine_out")
        self._in_dim = input_dim
        _ = self.call(tf.zeros((1, input_dim), dtype=tf.float32), training=False)

    def call(self, x, training=False):
        z = self.trunk(x, training=training)
        y = self.out(z)
        y = tf.reshape(y, (-1, 64, 8))
        return y


def build_sine_state_refiner(input_dim: int,
                             hidden=(256, 256),
                             w0: float = 1.0,
                             dropout: float = 0.0) -> SineStateRefiner:
    return SineStateRefiner(input_dim=input_dim, hidden=hidden, w0=w0, dropout=dropout)
