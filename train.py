import os
import dask
from atari_rl import create_runs, get_dask_client


BASE_DIR = "/content/repos/atari-rl"

LOG_DIR = f"{BASE_DIR}/agents"
CONFIGS_DIR = f"{BASE_DIR}/configs"
TENSORBOARD_DIR = f"{BASE_DIR}/logs/tensorboard"

WANDB_PROJECT_NAME = "Solen-RL-Project-2"
WANDB_ENTITY = "appliedtheta"

SEED = 43

SAVE_FREQ = 500000
EVAL_FREQ = 100000
EVAL_EPISODES = 5

DEVICE = "cuda"

ENVS_A = ["ALE/Breakout-v5",
          "ALE/SpaceInvaders-v5",
          "ALE/MsPacman-v5",
          "ALE/DonkeyKong-v5",
          "ALE/Frogger-v5"]

ENVS_B = ["ALE/Asteroids-v5",
          "ALE/Qbert-v5",
          "ALE/Pitfall-v5",
          "ALE/Centipede-v5",
          "ALE/Pong-v5"]

ALGOS = ["a2c"]

TRACK = True
RETURN_DELAYED = True

runs = create_runs(algos=ALGOS,
                   envs=ENVS_A,
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
                   track=TRACK,
                   return_delayed=RETURN_DELAYED)

if RETURN_DELAYED:

    with get_dask_client() as client:
        dask.compute(*runs)
