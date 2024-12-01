from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
import torch
import environment_pyboy_neat_mario as emt
import callback as cb
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO


def make_env(env_class):
    def _init():
        env = env_class()
        env = MaxAndSkipEnv(env, 15)
        return env
    return _init


def train_model(env_class, model_path, time_steps):
    n_envs = 20
    checkpoint_dir = './train/'
    log_dir = './logs/'
    env_class = emt.GbaGame  # Replace with your actual environment class
    # Create the vectorized environment
    # env = VecMonitor(SubprocVecEnv([make_env(env_class, i) for i in range(n_envs)]), "logs/TestMonitor")
    env = make_vec_env(make_env(env_class), n_envs=n_envs, monitor_dir=log_dir, vec_env_cls=SubprocVecEnv)
    # model_path = 'path_to_model'  # Replace with your actual model path

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    total_time_steps = time_steps
    callback = cb.TrainAndLoggingCallback(check_freq=4096, save_path=checkpoint_dir)

    policy_kwargs = dict(
        net_arch=[dict(pi=[128, 128], vf=[128, 128])]  # Example architecture: separate networks for policy and value
    )

    model = PPO('CnnPolicy', env, policy_kwargs=policy_kwargs, tensorboard_log=log_dir,
                verbose=1, gamma=0.999, n_steps=2048,
                n_epochs=10, batch_size=512, ent_coef=0.01, learning_rate=0.0003, device=device)
    if model_path:
        pass
        # model.load(model_path)

    model.learn(total_timesteps=total_time_steps, callback=callback)
    env.close()


if __name__ == '__main__':
    model_pth = 'train/end_of_training_run1.zip'
    train_model(emt.GbaGame, model_pth, time_steps=100000000)
