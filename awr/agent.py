import math
from collections import namedtuple

import numpy as np
import pyoneer as pynr
import tensorflow as tf
import tensorflow_probability as tfp


AgentPolicyOutput = namedtuple("AgentPolicyOutput", ["action"])
AgentValueOutput = namedtuple("AgentValueOutput", ["value"])
AgentPolicyValueOutput = namedtuple(
    "AgentPolicyValueOutput", ["log_prob", "entropy", "value"]
)


class Agent(tf.Module):
    def __init__(self, observation_space, action_space, state_spec, action_spec):
        super(Agent, self).__init__(name="Agent")

        # Parse input, output specs, shapes.
        self.observation_space = observation_space
        self.action_space = action_space
        self.state_spec = state_spec
        self.action_spec = action_spec
        self.output_specs = AgentPolicyOutput(action=self.action_spec)
        self.output_shapes = tf.nest.map_structure(
            lambda spec: spec.shape, self.output_specs
        )
        self.output_dtypes = tf.nest.map_structure(
            lambda spec: spec.dtype, self.output_specs
        )

        # Weights initializers.
        kernel_initializer = tf.initializers.VarianceScaling(scale=2.0)
        logits_initializer = tf.initializers.VarianceScaling(scale=1.0)

        # Discrete or continuous action space.
        if hasattr(action_space, "n"):
            self._is_discrete = True
            self._n_outputs = action_space.n
        else:
            self._is_discrete = False
            self._n_outputs = np.prod(action_space.shape)

        # Hidden layers.
        self._hidden = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    128, activation=tf.nn.relu, kernel_initializer=kernel_initializer
                ),
                tf.keras.layers.Dense(
                    64, activation=tf.nn.relu, kernel_initializer=kernel_initializer
                ),
            ]
        )

        # Output layer (logits or location, scale).
        if self._is_discrete:
            self._logits = tf.keras.layers.Dense(
                self._n_outputs, kernel_initializer=logits_initializer
            )
        else:
            self._loc = tf.keras.layers.Dense(
                units=self._n_outputs, kernel_initializer=logits_initializer
            )
            self._scale_diag = tf.keras.layers.Dense(
                units=self._n_outputs,
                activation=tf.exp,
                kernel_initializer=logits_initializer,
            )

        self._value = tf.keras.layers.Dense(1) 

    @property
    def value_trainable_variables(self):
        return self._hidden.trainable_variables + self._value.trainable_variables

    @property
    def policy_trainable_variables(self):
        if self._is_discrete:
            return self._hidden.trainable_variables + self._logits.trainable_variables
        else:
            return (
                self._hidden.trainable_variables
                + self._loc.trainable_variables
                + self._scale_diag.trainable_variables
            )

    def _scale_state(self, state):
        state = tf.cast(state, dtype=tf.float32)
        observation_high = np.where(
            self.observation_space.high < np.finfo(np.float32).max,
            self.observation_space.high,
            +1.0,
        )
        observation_low = np.where(
            self.observation_space.low > np.finfo(np.float32).min,
            self.observation_space.low,
            -1.0,
        )

        observation_mean, observation_var = pynr.moments.range_moments(
            observation_low, observation_high
        )
        state_norm = tf.math.divide_no_nan(
            state - observation_mean, tf.sqrt(observation_var)
        )
        return state_norm

    @tf.function
    def initialize(self, env_outputs, agent_outputs):
        state = self._scale_state(env_outputs.state)
        hidden = self._hidden(state)
        _ = self._value(hidden)

        if self._is_discrete:
            _ = self._logits(hidden)
        else:
            _ = self._loc(hidden)
            _ = self._scale_diag(hidden)

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

    def _discrete(self, hidden):
        logits = self._logits(hidden)
        policy = tfp.distributions.Categorical(logits=logits)
        return policy

    def _continuous(self, hidden):
        loc = self._loc(hidden)
        scale_diag = self._scale_diag(hidden)
        policy = tfp.distributions.Normal(loc=loc, scale=scale_diag)
        return policy

    @tf.function
    def policy_value(self, env_outputs, agent_outputs):
        state = env_outputs.state
        state = self._scale_state(state)
        hidden = self._hidden(state)

        if self._is_discrete:
            policy = self._discrete(hidden)
            entropy = policy.entropy()
        else:
            policy = self._continuous(hidden)
            entropy = -policy.log_prob(agent_outputs.action)
        
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

        if self._is_discrete:
            policy = self._discrete(hidden)
        else:
            policy = self._continuous(hidden)

        if explore:
            action = policy.sample()
        else:
            action = policy.mode()

        action = tf.nest.map_structure(
            lambda t, s: tf.cast(t, s.dtype), action, self.action_spec
        )
        return AgentPolicyOutput(action=action)