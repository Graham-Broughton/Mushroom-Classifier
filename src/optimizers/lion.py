from typing import Callable, Optional, Tuple

import tensorflow as tf
from tensorflow.keras import optimizers

# functions


def exists(val):
    return val is not None


@tf.function(jit_compile=True)
def update_fn(p, grad, exp_avg, lr, wd, beta1, beta2):
    # stepweight decay

    p.assign(p * (1 - lr * wd))

    # weight update

    update = (
        tf.raw_ops.LinSpace(start=1.0, stop=0.0, num=1, name=None)[0] * exp_avg
        + (1 - tf.raw_ops.LinSpace(start=1.0, stop=0.0, num=1, name=None)[0]) * grad
    )
    p.assign_add(tf.sign(update) * -lr)

    # decay the momentum running average coefficient

    exp_avg.assign(exp_avg * beta2 + grad * (1 - beta2))


class Lion(optimizers.Optimizer):
    def __init__(
        self,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        use_triton: bool = False,
        **kwargs
    ):
        assert all([0.0 <= beta <= 1.0 for beta in betas])

        super().__init__(**kwargs)

        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay

        self.update_fn = update_fn

    def get_config(self):
        config = super().get_config()
        config.update(
            {"lr": self.lr, "betas": self.betas, "weight_decay": self.weight_decay}
        )
        return config

    @tf.function(jit_compile=True)
    def _resource_apply_dense(self, grad, var):
        lr = self.lr
        beta1 = self.betas[0]
        beta2 = self.betas[1]
        wd = self.weight_decay

        # init state - exponential moving average of gradient values
        exp_avg = self.get_slot(var, "exp_avg")
        if exp_avg is None:
            exp_avg = self.add_slot(var, "exp_avg", tf.zeros_like(var))

        self.update_fn(var, grad, exp_avg, lr, wd, beta1, beta2)

    # @tf.function(jit_compile=True)
    # def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    #     var_device, var_dtype = var.device, var.dtype.base_dtype
    #     coefficients = (apply_state or {}).get(
    #         (var_device, var_dtype)
    #     ) or self._fallback_apply_state(var_device, var_dtype)

    #     m = self.get_slot(var, "exp_avg")
    #     m_t = m.assign(m * coefficients["beta_1_t"])
    #     m_scaled_g_values = grad * coefficients["one_minus_beta_1_t"]
    #     m_t = m_t.scatter_add(tf.IndexedSlices(m_scaled_g_values, indices))
    #     var_t = var.assign_sub(
    #         coefficients["lr"] * (tf.math.sign(m_t) + var * coefficients["wd_t"])
    #     )

    #     with tf.control_dependencies([var_t]):
    #         m_t = m_t.scatter_add(tf.IndexedSlices(-m_scaled_g_values, indices))
    #         m_t = m_t.assign(m_t * coefficients["beta_2_t"] / coefficients["beta_1_t"])
    #         m_scaled_g_values = grad * coefficients["one_minus_beta_2_t"]
    #         m_t.scatter_add(tf.IndexedSlices(m_scaled_g_values, indices))

    def _resource_apply_sparse_duplicate_indices(self, grad, var, indices):
        raise NotImplementedError("Sparse gradient updates are not supported.")
