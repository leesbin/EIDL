# Description: libray of Optimizer

import numpy as np

class AdamOptimizer:
        def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, warmup_iters=10):
            self.lr = lr
            self.beta1 = beta1
            self.beta2 = beta2
            self.epsilon = epsilon
            self.m = None
            self.v = None
            self.t = 0
            self.warmup_iters = warmup_iters

        def update(self, eps, grads):
            if self.m is None:
                self.m = np.zeros_like(eps)
            if self.v is None:
                self.v = np.zeros_like(eps)

            self.t += 1
            self.m = self.beta1 * self.m + (1 - self.beta1) * grads
            self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)

            m_hat = self.m / (1 - self.beta1 ** self.t)
            v_hat = self.v / (1 - self.beta2 ** self.t)

            # Warm-up 단계
            if False:
                if self.t <= self.warmup_iters:
                    warmup_factor = self.t / self.warmup_iters
                    lr = self.lr * warmup_factor
                else:
                    lr = self.lr
            else:
                lr = self.lr
            #update = self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            update_factor= m_hat / (np.sqrt(v_hat) + self.epsilon)
            update = lr * update_factor
            updated_eps = eps + update
            updated_eps = np.clip(updated_eps, 0.0, 1.0)

            adam_lr=np.mean(np.abs(update))
            adam_uf=np.mean(np.abs(update_factor))

            # adam_beta1=self.beta1
            # adam_beta2=self.beta2
            # adam_m=self.m
            # adam_v=self.v
            adam_t=self.t

            return updated_eps, adam_lr, adam_t, update_factor