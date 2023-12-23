import difflib
import importlib
import os
import time
import uuid
import gymnasium as gym
import numpy as np
import stable_baselines3 as sb3
import torch as th
import asyncio


from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path
from stable_baselines3.common.utils import set_random_seed

from rl_zoo3.exp_manager import ExperimentManager
from rl_zoo3.utils import ALGOS, StoreDict
from rl_zoo3.args import TrainArgs

from concurrent.futures import ThreadPoolExecutor


def call_train(run_idx: int,
               train_args: TrainArgs):
    """Train an RL agent for a given environment and hyperparameters."""

    # Going through custom gym packages to let them register in the global registory
    for env_module in train_args.gym_packages:
        importlib.import_module(env_module)

    env_id = train_args.env
    registered_envs = set(gym.envs.registry.keys())

    # If the environment is not found, suggest the closest match
    if env_id not in registered_envs:
        try:
            closest_match = difflib.get_close_matches(
                env_id, registered_envs, n=1)[0]
        except IndexError:
            closest_match = "'no close match found...'"
        raise ValueError(
            f"{env_id} not found in gym registry, you maybe meant {closest_match}?")

    # Unique id to ensure there is no race condition for the folder creation
    uuid_str = f"_{uuid.uuid4()}" if train_args.uuid_on else ""
    if train_args.seed < 0:
        # Seed but with a random one
        # type: ignore[attr-defined]
        train_args.seed = np.random.randint(2**32 - 1, dtype="int64").item()

    set_random_seed(train_args.seed)

    # Setting num threads to 1 makes things run faster on cpu
    if train_args.num_threads > 0:
        if train_args.verbose > 1:
            print(f"Setting torch.num_threads to {train_args.num_threads}")
        th.set_num_threads(train_args.num_threads)

    if train_args.trained_agent != "":
        assert train_args.trained_agent.endswith(".zip") and os.path.isfile(
            train_args.trained_agent
        ), "The trained_agent must be a valid path to a .zip file"

    print("=" * 10, env_id, "=" * 10)
    print(f"Seed: {train_args.seed}")

    if train_args.track:
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "if you want to use Weights & Biases to track experiment, please install W&B via `pip install wandb`"
            ) from e

        run_name = f"{train_args.env}__{train_args.algo}__{int(time.time())}"
        
        train_args.tensorboard_log = f"{train_args.tensorboard_log}/{run_name}"

        wandb.tensorboard.patch(root_logdir=train_args.tensorboard_log)

        run = wandb.init(name=run_name,
                         project=train_args.wandb_project_name,
                         entity=train_args.wandb_entity,
                         tags=train_args.wandb_tags,
                         config=vars(train_args),
                         sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                         monitor_gym=True,  # auto-upload the videos of agents playing the game
                         save_code=True)

    exp_manager = ExperimentManager(train_args,
                                    train_args.algo,
                                    env_id,
                                    train_args.log_folder,
                                    train_args.tensorboard_log,
                                    train_args.n_timesteps,
                                    train_args.eval_freq,
                                    train_args.eval_episodes,
                                    train_args.save_freq,
                                    train_args.hyperparams,
                                    train_args.env_kwargs,
                                    train_args.eval_env_kwargs,
                                    train_args.trained_agent,
                                    train_args.optimize_hyperparameters,
                                    train_args.storage,
                                    train_args.study_name,
                                    train_args.n_trials,
                                    train_args.max_total_trials,
                                    train_args.n_jobs,
                                    train_args.sampler,
                                    train_args.pruner,
                                    train_args.optimization_log_path,
                                    n_startup_trials=train_args.n_startup_trials,
                                    n_evaluations=train_args.n_evaluations,
                                    truncate_last_trajectory=train_args.truncate_last_trajectory,
                                    uuid_str=uuid_str,
                                    seed=train_args.seed,
                                    log_interval=train_args.log_interval,
                                    save_replay_buffer=train_args.save_replay_buffer,
                                    verbose=train_args.verbose,
                                    vec_env_type=train_args.vec_env,
                                    n_eval_envs=train_args.n_eval_envs,
                                    no_optim_plots=train_args.no_optim_plots,
                                    device=train_args.device,
                                    config=train_args.conf_file,
                                    show_progress=train_args.progress)

    # Prepare experiment and launch hyperparameter optimization if needed
    results = exp_manager.setup_experiment()
    if results is not None:
        model, saved_hyperparams = results
        if train_args.track:
            # we need to save the loaded hyperparameters
            train_args.saved_hyperparams = saved_hyperparams
            assert run is not None  # make mypy happy
            run.config.setdefaults(vars(train_args))

        # Normal training
        if model is not None:
            exp_manager.learn(model)
            exp_manager.save_trained_model(model)
    else:
        exp_manager.hyperparameters_optimization()


def async_train_2(run_idx: int,
                  train_args: TrainArgs):
    print(f"train -> run_idx: {run_idx}")
    time.sleep(5)
    return f"return -> run_idx: {run_idx}"


def train(n_runs: int,
          algos: List[str],
          envs: List[str],
          configs_dir: List[str],
          tensorboard_log: str = "",
          trained_agent: str = "",
          truncate_last_trajectory: bool = True,
          n_timesteps: int = -1,
          num_threads: int = -1,
          log_interval: int = -1,
          eval_freq: int = 25000,
          optimization_log_path: str = "",
          eval_episodes: int = 5,
          n_eval_envs: int = 1,
          save_freq: int = -1,
          save_replay_buffer: bool = False,
          log_folder: str = "logs",
          seed: int = -1,
          vec_env: str = "dummy",
          device: str = "auto",
          n_trials: int = 500,
          max_total_trials: int = -1,
          optimize_hyperparameters: bool = False,
          no_optim_plots: bool = False,
          n_jobs: int = 1,
          sampler: str = "tpe",
          pruner: str = "median",
          n_startup_trials: int = 10,
          n_evaluations: int = 1,
          storage: str = "",
          study_name: str = "",
          verbose: int = 1,
          gym_packages: List[str] = [],
          env_kwargs: Dict[str, str] = dict(),
          eval_env_kwargs: Dict[str, str] = dict(),
          hyperparams: Dict[str, str] = dict(),
          uuid_on: bool = False,
          track: bool = False,
          wandb_project_name: str = "sb3",
          wandb_entity: str = "",
          progress: bool = False):
    """Train an RL agent for a given environment and hyperparameters."""

    train_args = locals()

    n_runs = train_args.pop("n_runs")
    algos = train_args.pop("algos")
    envs = train_args.pop("envs")
    configs_dir = train_args.pop("configs_dir")

    called_futures = []

    with ThreadPoolExecutor(max_workers=n_runs) as executor:

        for run_idx in range(n_runs):

            algo = algos[run_idx]
            env = envs[run_idx]
            conf_file = Path(configs_dir) / f"{algo}.yml"

            _train_args = TrainArgs(algo=algo,
                                    env=env,
                                    conf_file=conf_file.as_posix(),
                                    wandb_tags=[algo, env],
                                    **train_args)

            called_futures.append(executor.submit(call_train,
                                                  run_idx=run_idx,
                                                  train_args=_train_args))

            time.sleep(10)

            print(f"------------------------------------\n\n")
            
    for called_future in called_futures:
        print("called_futrue: ", called_future.result())
