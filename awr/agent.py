import math
from collections import namedtuple

import pyoneer as pynr
import tensorflow as tf
import tensorflow_probability as tfp


AgentPolicyOutput = namedtuple("AgentPolicyOutput", ["action"])
AgentValueOutput = namedtuple("AgentValueOutput", ["value"])
AgentPolicyValueOutput = namedtuple(
    "AgentPolicyValueOutput", ["log_prob", "entropy", "value"]
)


class Agent(tf.Module):
    def __init__(self, action_spec):
        super(Agent, self).__init__(name="Agent")
        self._hidden = tf.keras.layers.Dense(32, activation=tf.nn.relu)
        self._logits = tf.keras.Sequential(
            [tf.keras.layers.Dense(8, activation=tf.nn.relu), tf.keras.layers.Dense(2)]
        )
        self._value = tf.keras.Sequential(
            [tf.keras.layers.Dense(8, activation=tf.nn.relu), tf.keras.layers.Dense(1)]
        )
        self._policy = tfp.distributions.Categorical

        self.action_spec = action_spec
        self.output_specs = AgentPolicyOutput(action=self.action_spec)
        self.output_shapes = tf.nest.map_structure(
            lambda spec: spec.shape, self.output_specs
        )
        self.output_dtypes = tf.nest.map_structure(
            lambda spec: spec.dtype, self.output_specs
        )

    @property
    def value_trainable_variables(self):
        return self._hidden.trainable_variables + self._value.trainable_variables

    @property
    def policy_trainable_variables(self):
        return self._hidden.trainable_variables + self._logits.trainable_variables

    @tf.function
    def _scale_state(self, state):
        state = tf.cast(state, tf.float32)
        state = state / [[2.4, 10.0, 1.0, 10.0]]
        state = tf.concat(
            [
                state,
                tf.stack(
                    [
                        tf.math.cos(state[..., 2] / math.pi),
                        tf.math.sin(state[..., 2] / math.pi),
                    ],
                    axis=-1,
                ),
            ],
            axis=-1,
        )
        return tf.clip_by_value(state, -1.0, 1.0)

    @tf.function
    def initialize(self, env_outputs, agent_outputs):
        state = self._scale_state(env_outputs.state)
        hidden = self._hidden(state)
        _ = self._value(hidden)
        _ = self._logits(hidden)

    @tf.function
    def value(self, env_outputs):
        state = self._scale_state(env_outputs.state)
        hidden = self._hidden(state)
        value = tf.squeeze(self._value(hidden), axis=-1)
        return AgentValueOutput(value=value)

    @tf.function
    def next_value(self, env_outputs):
        next_state = self._scale_state(env_outputs.next_state)
        hidden = self._hidden(next_state)
        value = tf.squeeze(self._value(hidden), axis=-1)
        return AgentValueOutput(value=value)

    @tf.function
    def policy_value(self, env_outputs, agent_outputs):
        state = self._scale_state(env_outputs.state)
        hidden = self._hidden(state)
        logits = self._logits(hidden)
        policy = self._policy(logits=logits)
        entropy = policy.entropy()
        log_prob = policy.log_prob(agent_outputs.action)
        value = tf.squeeze(self._value(hidden), axis=-1)
        return AgentPolicyValueOutput(log_prob=log_prob, entropy=entropy, value=value)

    @tf.function
    def reset(self, env_outputs, explore=True):
        initial_action = pynr.debugging.mock_spec(
            tf.TensorShape([env_outputs.state.shape[0]]), self.action_spec, tf.zeros
        )
        return AgentPolicyOutput(action=initial_action)

    @tf.function
    def step(self, env_outputs, agent_outputs, time_step, explore=True):
        state = env_outputs.next_state
        state = self._scale_state(state)
        hidden = self._hidden(state)
        logits = self._logits(hidden)
        policy = self._policy(logits=logits)

        if explore:
            action = policy.sample()
        else:
            action = policy.mode()

        action = tf.nest.map_structure(
            lambda t, s: tf.cast(t, s.dtype), action, self.action_spec
        )
        return AgentPolicyOutput(action=action)
