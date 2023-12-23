import os
from utils.train import train

BASE_DIR = "/home/sequenzia/dev/atari-rl"

LOG_DIR = f"{BASE_DIR}/trained-agents"
CONFIGS_DIR = f"{BASE_DIR}/configs"
TENSORBOARD_DIR = f"{BASE_DIR}/logs/tensorboard"

SEED = 43

N_TIMESTEMPS = 100000

SAVE_FREQ = 100000
EVAL_FREQ = 10000
EVAL_EPISODES = 5

PROJECT_NAME = "Solen-RL-Project-2"

os.environ["WANDB_API_KEY"] = "8c880e6018cf423b7714cf055c5fd6152e1ae117"
os.environ["WANDB_DIR"] = f"{BASE_DIR}/logs"

train(n_runs=8,
      algos=["ppo",
             "a2c",
             "ppo",
             "a2c",
             "ppo",
             "a2c",
             "ppo",
             "a2c"],
      envs=["BreakoutNoFrameskip-v4",
            "BreakoutNoFrameskip-v4",
            "PongNoFrameskip-v4",
            "PongNoFrameskip-v4",
            "SpaceInvadersNoFrameskip-v4",
            "SpaceInvadersNoFrameskip-v4",
            "MsPacmanNoFrameskip-v4",
            "MsPacmanNoFrameskip-v4"],
      n_timesteps=N_TIMESTEMPS,
      log_folder=LOG_DIR,
      configs_dir=CONFIGS_DIR,
      tensorboard_log=TENSORBOARD_DIR,
      seed=SEED,
      save_freq=SAVE_FREQ,
      eval_freq=EVAL_FREQ,
      eval_episodes=EVAL_EPISODES,
      wandb_project_name=PROJECT_NAME,
      wandb_entity="appliedtheta",
      device="cuda",
      track=True)
