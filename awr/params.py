import attr
import json
import tensorflow as tf


@attr.s
class HyperParams:
    seed = attr.ib(type=int)
    env = attr.ib(type=str)
    learning_rate = attr.ib(default=1e-3, type=float)
    flatten = attr.ib(default=False, type=bool)  # flatten episode/step dims
    iterations = attr.ib(default=1000, type=int)  # number of train/eval iterations
    max_size = attr.ib(default=int(5e2), type=int)  # maximum transition buffer size
    max_size_flat = attr.ib(default=int(5e4))
    num_samples = attr.ib(default=64 * 100, type=int)  # number of transitions to sample
    num_samples_flat = attr.ib(
        default=256 * 250, type=int
    )  # number of transitions to sample
    batch_size = attr.ib(default=64, type=int)  # dataset batch size each iteration
    batch_size_flat = attr.ib(
        default=256, type=int
    )  # dataset batch size each iteration
    steps_init = attr.ib(default=int(1e2), type=int)
    steps_init_flat = attr.ib(default=int(1e3), type=int)  # number of initial episodes
    eval_iters = attr.ib(default=10, type=int)
    episodes_train = attr.ib(default=10, type=int)  # number of episodes to append
    episodes_eval = attr.ib(default=25, type=int)  # number of evaluation episodes
    discount = attr.ib(default=0.99, type=float)
    clipnorm = attr.ib(default=1.0, type=float)
    value_scale = attr.ib(default=0.5, type=float)
    lambda_ = attr.ib(default=0.95, type=float)
    beta = attr.ib(default=0.01, type=float)
    score_max = attr.ib(default=20.0, type=float)
    l2_coef = attr.ib(default=1e-3, type=float)

    @staticmethod
    def load(path):
        with tf.io.gfile.GFile(path, mode="r") as fp:
            data = json.load(fp)
        params = HyperParams(**data)
        return params

    def save(self, path):
        with tf.io.gfile.GFile(path, mode="w") as fp:
            json.dump(attr.asdict(self), fp)
