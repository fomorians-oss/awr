import os
import time
import random
import argparse
import numpy as np
import tensorflow as tf

from awr.params import HyperParams
from awr.algorithm import Algorithm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", required=True, type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--env", default="CartPole-v0")
    parser.add_argument("--flatten", action="store_true")
    parser.add_argument("--gcp", action="store_true")
    args = parser.parse_args()
    print("args:", args)

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # hyperparams
    params_path = os.path.join(args.job_dir, "params.json")
    params = HyperParams(seed=args.seed, env=args.env, flatten=args.flatten)
    params.save(params_path)
    print("params:", params)

    # training
    algorithm = Algorithm(job_dir=args.job_dir, params=params, gcp=args.gcp)
    algorithm.eval()


if __name__ == "__main__":
    main()
