import os
import gym

#from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines.common.env_checker import check_env
from stable_baselines.gail.dataset.dataset import ExpertDataset

from topoenv import TopoEnv
from policynet_v2 import GnnPolicy

MODEL_NAME = "gnnppo4topo_pretrain"
TENSORBOARD_LOG_DIR =  'tensorlog_topo'
TRAINSTEPS = 320000
HAS_PRETRAINED = False

def env_fn():
    env = TopoEnv()
    return env

def main():

    if not os.path.exists(TENSORBOARD_LOG_DIR):
        os.makedirs(TENSORBOARD_LOG_DIR)

    env_fns = []
    for _ in range(32):
        env_fns.append(env_fn)

    vec_env = DummyVecEnv(env_fns)

    if HAS_PRETRAINED:
        model = PPO2.load("gnn_ppo4topo8",vec_env)
    else:
        model = PPO2(GnnPolicy, vec_env, gamma=1, n_steps=4, verbose=1, tensorboard_log=TENSORBOARD_LOG_DIR)

    print("Model built.")
    dataset = ExpertDataset(expert_path="./pretraindata.npz",batch_size=32)
    model.pretrain(dataset)
    print("Pretraining terminated.")

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