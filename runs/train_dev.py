import os
import dask
from atari_rl import create_runs, get_dask_client

BASE_DIR = "/home/sequenzia/dev/repos/atari-rl"

LOG_DIR = f"{BASE_DIR}/agents"
CONFIGS_DIR = f"{BASE_DIR}/configs"
TENSORBOARD_DIR = f"{BASE_DIR}/logs/tensorboard"

WANDB_PROJECT_NAME = "Solen-RL-Project-2"
WANDB_ENTITY = "appliedtheta"

DEVICE = "cuda"

SEED = 43

N_TIMESTEMPS = 10000

SAVE_FREQ = 10000
EVAL_FREQ = 10000
EVAL_EPISODES = 5

# OPTIMIZE_ON = True
# N_TRIALS = 1000
# N_JOBS = 2
# SAMPLER = "tpe"
# PRUNER = "median"


# ALGOS = ["ppo", "a2c"]

# ENVS = ["BreakoutNoFrameskip-v4",
#          "PongNoFrameskip-v4",
#          "SpaceInvadersNoFrameskip-v4",
#          "MsPacmanNoFrameskip-v4"]

ALGOS = ["her"]

ENVS = ["BreakoutNoFrameskip-v4"]

RETURN_DELAYED = False

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
                   wandb_project_name=WANDB_PROJECT_NAME,
                   wandb_entity=WANDB_ENTITY,
                   device=DEVICE,
                #    track=True,
                   return_delayed=RETURN_DELAYED)

if RETURN_DELAYED:

    with get_dask_client() as client:
        dask.compute(*runs)