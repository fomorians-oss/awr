import attr
import json
import tensorflow as tf


@attr.s
class HyperParams:
    seed = attr.ib(type=int)
    env = attr.ib(type=str)
    learning_rate = attr.ib(default=1e-3, type=float)
    flatten = attr.ib(default=False, type=bool)  # flatten episode/step dims
    num_critics = attr.ib(default=2, type=int)
    iterations = attr.ib(default=1000, type=int)  # number of train/eval iterations
    max_size = attr.ib(default=int(1e6), type=int)  # maximum transition buffer size
    num_value_samples = attr.ib(
        default=256 * 100, type=int
    )  # number of transitions to sample for value function training
    num_policy_samples = attr.ib(
        default=256 * 100, type=int
    )  # number of transitions to sample for policy training
    batch_size = attr.ib(default=4, type=int)  # dataset batch size each iteration
    env_batch_size = attr.ib(default=100, type=int)
    steps_init = attr.ib(default=int(1e3), type=int)  # number of initial episodes
    eval_iters = attr.ib(default=10, type=int)
    episodes_train = attr.ib(default=1, type=int)  # number of episodes to append
    episodes_eval = attr.ib(default=100, type=int)  # number of evaluation episodes
    discount = attr.ib(default=0.99, type=float)
    grad_clipping = attr.ib(default=1.0, type=float)
    value_scale = attr.ib(default=0.5, type=float)
    lambda_ = attr.ib(default=0.95, type=float)
    beta = attr.ib(default=0.05, type=float)
    score_max = attr.ib(default=20.0, type=float)

    @staticmethod
    def load(path):
        with tf.io.gfile.GFile(path, mode="r") as fp:
            data = json.load(fp)
        params = HyperParams(**data)
        return params

    def save(self, path):
        with tf.io.gfile.GFile(path, mode="w") as fp:
            json.dump(attr.asdict(self), fp)
