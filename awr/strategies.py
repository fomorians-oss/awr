import tensorflow as tf


class Strategy:
    def __init__(self, agent, explore):
        self.agent = agent
        self.explore = explore

    @tf.function
    def reset(self, *args, **kwargs):
        return self.agent.reset(*args, explore=self.explore, **kwargs)

    @tf.function
    def step(self, *args, **kwargs):
        return self.agent.step(*args, explore=self.explore, **kwargs)
