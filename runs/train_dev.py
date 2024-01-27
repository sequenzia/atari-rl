import os
from atari_rl import create_runs

BASE_DIR = "/home/sequenzia/dev/repos/atari-rl"

LOG_DIR = f"{BASE_DIR}/agents"
CONFIGS_DIR = f"{BASE_DIR}/configs"
TENSORBOARD_DIR = f"{BASE_DIR}/logs/tensorboard"

SEED = 43

N_TIMESTEMPS = 10000

SAVE_FREQ = 100000
EVAL_FREQ = 10000
EVAL_EPISODES = 5

PROJECT_NAME = "Solen-RL-Project-2"

os.environ["WANDB_API_KEY"] = "8c880e6018cf423b7714cf055c5fd6152e1ae117"
os.environ["WANDB_DIR"] = f"{BASE_DIR}/logs"

# ALGOS = ["ppo", "a2c"]

# ENVS = ["BreakoutNoFrameskip-v4",
#          "PongNoFrameskip-v4",
#          "SpaceInvadersNoFrameskip-v4",
#          "MsPacmanNoFrameskip-v4"]

ALGOS = ["ppo"]

ENVS = ["BreakoutNoFrameskip-v4"]

runs = create_runs(algos=ALGOS,
                   envs=ENVS,
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
