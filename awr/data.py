from collections import namedtuple

import numpy as np
import tensorflow as tf

from awr.agent import AgentPolicyOutput
from pyoneer.rl.rollouts.gym_ops import Transition


def episodic_mean(inputs):
    return tf.reduce_mean(tf.reduce_sum(inputs, axis=1))


def flatten_transitions(agent_outputs, env_outputs):
    # flatten episode and step dimensions of transitions
    agent_outputs = agent_outputs._asdict()
    env_outputs = env_outputs._asdict()

    agent_outputs_flat = {}
    for key, val in agent_outputs.items():
        agent_outputs_flat[key] = tf.reshape(
            val, shape=(np.prod(val.shape[0:2]), 1) + val.shape[2:]
        )

    env_outputs_flat = {}
    for key, val in env_outputs.items():
        env_outputs_flat[key] = tf.reshape(
            val, shape=(np.prod(val.shape[0:2]), 1) + val.shape[2:]
        )

    indices = env_outputs_flat["weight"] > 0.0
    indices = tf.squeeze(indices)

    # remove empty transitions
    for key, val in agent_outputs_flat.items():
        agent_outputs_flat[key] = val[indices]

    for key, val in env_outputs_flat.items():
        env_outputs_flat[key] = val[indices]

    agent_outputs = AgentPolicyOutput(action=agent_outputs_flat["action"])
    env_outputs = Transition(
        state=env_outputs_flat["state"],
        reward=env_outputs_flat["reward"],
        next_state=env_outputs_flat["next_state"],
        terminal=env_outputs_flat["terminal"],
        weight=env_outputs_flat["weight"],
    )

    return agent_outputs, env_outputs
