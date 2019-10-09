import os
import sys
import gym
import numpy as np
import pyoneer as pynr
import pyoneer.rl as pyrl
import tensorflow as tf

from awr.agent import Agent
from awr.strategies import Strategy
from awr.env import create_env_model
from awr.data import episodic_mean, flatten_transitions


class Algorithm:
    def __init__(self, job_dir, params, data_dir=None):
        self.job_dir = job_dir
        self.params = params
        self.data_dir = data_dir

        self.explore_env_model = create_env_model(params.env, params.episodes_train)
        self.exploit_env_model = create_env_model(params.env, params.episodes_eval)

        self.explore_env_model.seed(params.seed)
        self.exploit_env_model.seed(params.seed)

        self.max_steps = self.exploit_env_model._max_episode_steps

        self.agent = Agent(
            self.explore_env_model._env.action_space,
            self.explore_env_model.state_spec,
            self.explore_env_model.action_spec,
        )
        self.value_optimizer = tf.keras.optimizers.Adam(
            params.learning_rate, clipnorm=params.grad_clipping
        )
        self.policy_optimizer = tf.keras.optimizers.Adam(
            params.learning_rate, clipnorm=params.grad_clipping
        )

        if params.flatten:
            mock_env_outputs = pynr.debugging.mock_spec(
                tf.TensorShape([1]), self.explore_env_model.output_specs, tf.zeros
            )
            mock_agent_outputs = pynr.debugging.mock_spec(
                tf.TensorShape([1]), self.agent.output_specs, tf.zeros
            )
        else:
            mock_env_outputs = pynr.debugging.mock_spec(
                tf.TensorShape([1, self.max_steps]),
                self.explore_env_model.output_specs,
                tf.zeros,
            )
            mock_agent_outputs = pynr.debugging.mock_spec(
                tf.TensorShape([1, self.max_steps]), self.agent.output_specs, tf.zeros
            )

        self.agent.initialize(
            env_outputs=mock_env_outputs, agent_outputs=mock_agent_outputs
        )

        if params.flatten:
            n_step = 1
        if not params.flatten:
            n_step = self.max_steps
            params.num_value_samples //= 100
            params.num_policy_samples //= 100
            params.max_size //= self.max_steps
            params.steps_init //= 100

        # Instantiate replay buffer.
        self.buffer = pyrl.transitions.ReplayBuffer(
            (self.agent.output_specs, self.exploit_env_model.output_specs),
            n_step=n_step,
            max_size=params.max_size,
        )

        # Load batch dataset if directory specified.
        if data_dir is not None:
            self.online_job = pynr.jobs.Job(directory=data_dir, buffer=self.buffer)
            status = self.online_job.restore().expect_partial()

        checkpointables = {
            "agent": self.agent,
            "value_optimizer": self.value_optimizer,
            "policy_optimizer": self.policy_optimizer,
            "buffer": self.buffer,
        }

        self.job = pynr.jobs.Job(directory=job_dir, max_to_keep=5, **checkpointables)

        self.discounted_returns = tf.function(pyrl.targets.discounted_returns)
        self.generalized_advantage_estimate = tf.function(
            pyrl.targets.generalized_advantage_estimate
        )

        self.explore_strategy = Strategy(self.agent, explore=True)
        self.exploit_strategy = Strategy(self.agent, explore=False)

        self.explore_rollout = pyrl.rollouts.Rollout(
            env=self.explore_env_model,
            agent=self.explore_strategy,
            n_step=self.max_steps,
        )
        self.exploit_rollout = pyrl.rollouts.Rollout(
            env=self.exploit_env_model,
            agent=self.exploit_strategy,
            n_step=self.max_steps,
        )

    @tf.function
    def _train_value(self, env_outputs):
        # Get bootstrap values if training on one-step rollouts.
        bootstrap_value = None
        if self.params.flatten:
            next_value = self.agent.next_value(env_outputs).value
            bootstrap_value = next_value * (
                1 - tf.cast(env_outputs.terminal, tf.float32)
            )
            bootstrap_value = tf.squeeze(bootstrap_value)

        # Compute discounted returns to use as value network regression targets.
        returns = self.discounted_returns(
            tf.cast(env_outputs.reward, tf.float32) * env_outputs.weight,
            bootstrap_value=bootstrap_value,
            discounts=self.params.discount,
            weights=env_outputs.weight,
        )

        with tf.GradientTape() as tape:
            # Compute value of states in sampled trajectories.
            agent_value_outputs = self.agent.value(env_outputs)

            # Compute value loss as mean squared error
            # between predicted values and actual returns.
            value_loss = self.params.value_scale * tf.reduce_sum(
                (
                    tf.square(agent_value_outputs.value - tf.stop_gradient(returns))
                    * env_outputs.weight
                )
            )

            if self.params.flatten:
                loss = value_loss / self.params.batch_size
            else:
                loss = value_loss / (self.params.batch_size * self.max_steps)

        # Compute and apply value network gradients.
        variables = self.agent.value_trainable_variables
        grads = tape.gradient(loss, variables)
        self.value_optimizer.apply_gradients(zip(grads, variables))

        # Make summaries.
        with self.job.summary_context("train"):
            grad_norm = tf.linalg.global_norm(grads)

            tf.summary.histogram(
                "q_values",
                agent_value_outputs.value,
                step=self.value_optimizer.iterations,
            )
            tf.summary.scalar(
                "losses/critic", loss, step=self.value_optimizer.iterations
            )
            tf.summary.scalar(
                "grad_norm/critic", grad_norm, step=self.value_optimizer.iterations
            )

    @tf.function
    def _train_policy(self, agent_outputs, env_outputs):
        # Compute value baseline.
        agent_value_outputs = self.agent.value(env_outputs)

        # Get bootstrap values if training on one-step rollouts.
        bootstrap_value = None
        if self.params.flatten:
            next_value = self.agent.next_value(env_outputs).value
            bootstrap_value = next_value * (
                1 - tf.cast(env_outputs.terminal, tf.float32)
            )
            bootstrap_value = tf.squeeze(bootstrap_value)

        # Compute advantages using TD(lambda).
        advantages = self.generalized_advantage_estimate(
            tf.cast(env_outputs.reward, tf.float32) * env_outputs.weight,
            agent_value_outputs.value * env_outputs.weight,
            discounts=self.params.discount,
            lambdas=self.params.lambda_,
            weights=env_outputs.weight,
            last_value=bootstrap_value,
        )

        with tf.GradientTape() as tape:
            # Compute estimate of policy distribution.
            agent_estimates_output = self.agent.policy_value(env_outputs, agent_outputs)

            # Compute unnormalized distribution implied by scaled, exponentiated advantages.
            score = tf.minimum(
                tf.exp(advantages / self.params.beta), self.params.score_max
            )

            # Compute policy loss as mismatch between policy and scaled advantage distribution.
            policy_loss = -tf.reduce_sum(
                agent_estimates_output.log_prob
                * tf.stop_gradient(score)
                * env_outputs.weight
            )
            loss = policy_loss / (self.params.batch_size * self.max_steps)

        # Compute and apply gradients to policy parameters.
        variables = self.agent.policy_trainable_variables
        grads = tape.gradient(loss, variables)
        self.policy_optimizer.apply_gradients(zip(grads, variables))

        entropy = tf.reduce_mean(agent_estimates_output.entropy)
        value = tf.reduce_mean(agent_estimates_output.value)
        log_prob = tf.reduce_mean(agent_estimates_output.log_prob)

        # Make summaries.
        with self.job.summary_context("train"):
            grad_norm = tf.linalg.global_norm(grads)

            tf.summary.scalar(
                "losses/policy", loss, step=self.policy_optimizer.iterations
            )
            tf.summary.scalar(
                "grad_norm/policy", grad_norm, step=self.policy_optimizer.iterations
            )
            tf.summary.scalar(
                "policy/entropy", entropy, step=self.policy_optimizer.iterations
            )
            tf.summary.scalar(
                "policy/value", value, step=self.policy_optimizer.iterations
            )
            tf.summary.scalar(
                "policy/log_prob", log_prob, step=self.policy_optimizer.iterations
            )

    @tf.function
    def _collect_transitions(self, policy, episodes):
        # Collect new transitions using the exploration policy.
        with tf.device("/cpu:0"):
            agent_outputs, env_outputs = self.explore_rollout().outputs

        return agent_outputs, env_outputs

    @tf.function
    def _update_buffer(self, agent_outputs, env_outputs):
        # Add new transitions to replay buffer.
        if self.params.flatten:
            self.buffer.write(flatten_transitions(agent_outputs, env_outputs))
        else:
            self.buffer.write((agent_outputs, env_outputs))
        
        # Make summaries.
        with self.job.summary_context("train"):
            tf.summary.scalar(
                "buffer/size", self.buffer.count, step=self.policy_optimizer.iterations
            )

    def _train(self, it):
        # Data collection.
        if self.data_dir is None:
            if self.params.episodes_train > 0:
                # Collect new transitions using exploration policy.
                agent_outputs, env_outputs = self._collect_transitions(
                    self.explore_strategy, self.params.episodes_train
                )

                # Update transition buffer with exploration trajectories.
                self._update_buffer(agent_outputs, env_outputs)

                # Make summaries.
                with self.job.summary_context("train"):
                    episodic_reward = episodic_mean(env_outputs.reward)

                    tf.summary.scalar(
                        "episodic_rewards/train",
                        episodic_reward,
                        step=self.policy_optimizer.iterations,
                    )

        # Sample trajectories and train value network.
        _, env_outputs = self.buffer.sample(size=self.params.num_value_samples)
        dataset = (
            tf.data.Dataset.from_tensor_slices(env_outputs)
            .batch(self.params.batch_size, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )
        for env_outputs in dataset:
            self._train_value(env_outputs)

        # Sample trajectories and train policy network.
        agent_outputs, env_outputs = self.buffer.sample(size=self.params.num_policy_samples)

        dataset = (
            tf.data.Dataset.from_tensor_slices((agent_outputs, env_outputs))
            .batch(self.params.batch_size, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )
        for agent_outputs, env_outputs in dataset:
            self._train_policy(agent_outputs, env_outputs)
            
        tf.print("Iteration %d" % it)

    def _eval(self, it):
        # Run rollouts under exploitation policy to evaluate performance.
        _, eval_env_outputs = self.exploit_rollout().outputs
        eval_returns = tf.reduce_sum(
            tf.cast(eval_env_outputs.reward, tf.float32) * eval_env_outputs.weight,
            axis=1,
        )
        episodic_reward = tf.reduce_mean(eval_returns)

        # Make summaries
        with self.job.summary_context("eval"):
            tf.summary.scalar(
                "episodic_rewards/eval",
                episodic_reward,
                step=self.policy_optimizer.iterations,
            )

        tf.print("Avg. eval reward: %.3f" % episodic_reward.numpy())

    def _train_iter(self, it):
        # Train.
        with pynr.debugging.Stopwatch() as stopwatch:
            self._train(it)

        with self.job.summary_context("train"):
            tf.summary.scalar(
                "time/train", stopwatch.duration, step=self.policy_optimizer.iterations
            )

        # Evaluate.
        if it % self.params.eval_iters == self.params.eval_iters - 1:
            with pynr.debugging.Stopwatch() as stopwatch:
                if self.params.episodes_eval > 0:
                    self._eval(it)

            with self.job.summary_context("eval"):
                tf.summary.scalar(
                    "time/eval",
                    stopwatch.duration,
                    step=self.policy_optimizer.iterations,
                )

            self.job.save(checkpoint_number=self.policy_optimizer.iterations)
            self.job.flush_summaries()

        sys.stdout.flush()
        sys.stderr.flush()

    def train(self):
        # Sample random trajectories to pre-fill the replay buffer.
        if self.data_dir is None:
            while self.buffer.count < self.params.steps_init:
                agent_outputs, env_outputs = self._collect_transitions(
                    policy=self.explore_strategy, episodes=1
                )
                self._update_buffer(agent_outputs, env_outputs)

        # Begin training.
        for it in range(self.params.iterations):
            with pynr.debugging.Stopwatch() as stopwatch:
                self._train_iter(it)
