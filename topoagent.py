import os
import gym

#from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines.common.env_checker import check_env

from topoenv import TopoEnv
from policynet_v2 import GnnPolicy

MODEL_NAME = "gnnppo4topo_new1"
TENSORBOARD_LOG_DIR =  'tensorlog_topo'
TRAINSTEPS = 100000
HAS_PRETRAINED = False

def env_fn():
    env = TopoEnv()
    return env

def main():

    if not os.path.exists(TENSORBOARD_LOG_DIR):
        os.makedirs(TENSORBOARD_LOG_DIR)

    env_fns = []
    for _ in range(2):
        env_fns.append(env_fn)

    vec_env = DummyVecEnv(env_fns)

    if HAS_PRETRAINED:
        model = PPO2.load("gnnppo4topo_pretrain",vec_env)
    else:
        model = PPO2(GnnPolicy, vec_env, gamma=1,verbose=1, tensorboard_log=TENSORBOARD_LOG_DIR)

    print("Model built.")
    model.learn(total_timesteps=TRAINSTEPS)
    print("Training terminated.")

    # save model
    model.save(MODEL_NAME)

    """
    # Test the model
    obs = env.reset()
    stop = False
    rewards = 0
    while not stop:
        action, _states = model.predict(obs)
        obs, reward, stop = env.step(action)
        rewards += reward
    print("Reward: ", rewards)
    """
    
if __name__ == "__main__":
    main()