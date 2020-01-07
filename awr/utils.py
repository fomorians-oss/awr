import tensorflow as tf

from pyoneer.rl.rollouts.gym_ops import Transition


def compute_return(reward, weight):
    """
    Computes undiscounted cumulative sum of rewards over time.

    :param reward: Tensor of rewards of shape ``[batch_size, time]``.
    :param weights: Boolean tensor specifying to valid transitions.
    :return: Undiscounted cumulative sum of rewards over time.
    """
    return tf.reduce_sum(reward * weight, axis=1)


def compute_ris_estimate(behavioral_agent, evaluation_agent, dataset):
    """
    Compute Regression Importance Sampling (RIS) (https://arxiv.org/abs/1806.01347)
    estimate. Re-weights returns by ratio of probability of trajectory under evaluation
    policy over probability of trajectory under behavioral policy.

    :param behavioral_agent: Behavioral policy network.
    :param evaluation_agent: Evaluation policy network.
    :param dataset: Tensorflow Dataset include states, actions, and rewards.
    :return: Estimate of return under evaluation policy.
    """
    behavioral_probs = []
    evaluation_probs = []
    ris_estimates = []
    for action, state, reward, next_state, terminal, weight in dataset:
        env_outputs = Transition(
            state=state,
            reward=reward,
            next_state=next_state,
            terminal=terminal,
            weight=weight,
        )

        # Compute return of trajectory.
        rtrn = compute_return(reward, weight)

        # Compute probability of transitions in trajectory
        # under evalaution and behavioral policies.
        behavioral_prob = behavioral_agent.policy(env_outputs).prob(action)
        evaluation_prob = evaluation_agent.policy(env_outputs).prob(action)

        # Compute odds (evaluation policy probs divided by behavorial policy probs).
        odds = tf.math.divide_no_nan(evaluation_prob, behavioral_prob)
        odds = tf.where(tf.equal(weight, 0), tf.ones_like(weight), odds)

        # Compute product of probabilities over whole trajectories.
        ris_weight = tf.reduce_sum(odds, axis=1)

        # Re-weight
        ris_estimate = ris_weight * rtrn

        behavioral_probs.append(behavioral_prob)
        evaluation_probs.append(evaluation_prob)
        ris_estimates.append(ris_estimate)
    
    return tf.reduce_mean(ris_estimates)
