import tensorflow as tf


class WarmupExponentialDecay(tf.keras.optimizers.schedules.ExponentialDecay):
    def __init__(self, warmup_steps=0, cold_steps=43700, cold_factor=0.1, final_learning_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.warmup_steps = warmup_steps
        self.cold_steps = cold_steps
        self.cold_factor = cold_factor
        self.final_learning_rate = final_learning_rate

    @tf.function
    def __call__(self, step):
        return tf.where(step <= self.cold_steps + self.warmup_steps, 
                        tf.where(step <= self.cold_steps, 
                                 self.initial_learning_rate*self.cold_factor,
                                 self.initial_learning_rate*(self.cold_factor + tf.cast(step - self.cold_steps, tf.float32)*(1 - self.cold_factor)/tf.cast(self.warmup_steps, tf.float32))), 
                        tf.maximum(super().__call__(step - self.cold_steps - self.warmup_steps), self.final_learning_rate))

    def get_config(self):
        config = super().get_config()
        config.update({'warmup_steps': self.warmup_steps})
        config.update({'cold_steps': self.cold_steps})
        config.update({'cold_factor': self.cold_factor})
        config.update({'final_learning_rate': self.final_learning_rate})
        return config
