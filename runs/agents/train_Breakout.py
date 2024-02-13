import os
import dask
from atari_rl import create_runs, get_dask_client

BASE_DIR = "/home/ubuntu/dev/repos/atari-rl"

LOG_DIR = f"{BASE_DIR}/agents"
CONFIGS_DIR = f"{BASE_DIR}/configs"
TENSORBOARD_DIR = f"{BASE_DIR}/logs/tensorboard"

WANDB_PROJECT_NAME = "Solen-Project"
WANDB_ENTITY = "appliedtheta"

SEED = 43

SAVE_FREQ = 500000
EVAL_FREQ = 100000
EVAL_EPISODES = 5

DEVICE = "cuda"
NUM_THREADS = 1
PLATFORM = "Lambda"

GAMES = ["Breakout"]

ALGOS = ["dqn"]

TRAINED_AGENT = "/home/ubuntu/dev/repos/atari-rl/agents/dqn/ALE-Breakout-v5_1/rl_model_10500000_steps.zip"

TRACK = True
DASK_ON = False

runs = create_runs(algos=ALGOS,
                   games=GAMES,
                   trained_agent=TRAINED_AGENT,
                   log_folder=LOG_DIR,
                   configs_dir=CONFIGS_DIR,
                   tensorboard_log=TENSORBOARD_DIR,
                   seed=SEED,
                   save_freq=SAVE_FREQ,
                   eval_freq=EVAL_FREQ,
                   eval_episodes=EVAL_EPISODES,
                   wandb_project_name=WANDB_PROJECT_NAME,
                   wandb_entity=WANDB_ENTITY,
                   device=DEVICE,
                   num_threads=NUM_THREADS,
                   platform=PLATFORM,
                   track=TRACK,
                   dask_on=DASK_ON)

if DASK_ON:

    with get_dask_client() as client:
        dask.compute(*runs)
