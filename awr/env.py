import gym
import pyoneer.rl as pyrl


def create_env_model(env="CartPole-v1", batch_size=None):
    """Create the gym env, wrapped in a vectorized manner."""
    env_spec = env

    def make_fn():
        env = gym.make(env_spec)
        env.observation_space.dtype = "float64"
        return env

    if batch_size is None:
        gym_env = make_fn()
    else:
        gym_env = pyrl.wrappers.Batch(make_fn, batch_size)

    # Wrap it in a pyrl.rollouts.Env.
    env_model = pyrl.rollouts.Env(gym_env)
    return env_model
