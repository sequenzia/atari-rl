from __future__ import annotations

import difflib
import importlib
import os
import time
import uuid
import gymnasium as gym
import numpy as np
import stable_baselines3 as sb3
import torch as th

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path
from stable_baselines3.common.utils import set_random_seed

from rl_zoo3.exp_manager import ExperimentManager
from rl_zoo3.utils import ALGOS
from rl_zoo3.args import TrainArgs


def create_runs(algos: List[str],
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

    local_args = locals()

    algos = local_args.pop("algos")
    envs = local_args.pop("envs")
    configs_dir = local_args.pop("configs_dir")

    runs = []

    run_idx = 0
    for algo in algos:

        for env in envs:

            conf_file = Path(configs_dir) / f"{algo}.yml"

            print(f"----------- {run_idx} -> {algo} | {env} -----------\n")

            train_args = TrainArgs(algo=algo,
                                   env=env,
                                   conf_file=conf_file.as_posix(),
                                   wandb_tags=[algo, env],
                                   **local_args)

            run = TrainRun(run_idx=run_idx,
                           train_args=train_args)

            runs.append(run)

            run_idx += 1

    return runs


@dataclass
class TrainRun:

    run_idx: int
    train_args: TrainArgs
    exp_manager: Optional[ExperimentManager] = field(default=None)
    model: Optional[sb3.base.BaseAlgorithm] = field(default=None)

    def train(self):

        # Going through custom gym packages to let them register in the global registory
        for env_module in self.train_args.gym_packages:
            importlib.import_module(env_module)

        env_id = self.train_args.env
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
        uuid_str = f"_{uuid.uuid4()}" if self.train_args.uuid_on else ""
        if self.train_args.seed < 0:
            # Seed but with a random one
            # type: ignore[attr-defined]
            self.train_args.seed = np.random.randint(
                2**32 - 1, dtype="int64").item()

        set_random_seed(self.train_args.seed)

        # Setting num threads to 1 makes things run faster on cpu
        if self.train_args.num_threads > 0:
            if self.train_args.verbose > 1:
                print(
                    f"Setting torch.num_threads to {self.train_args.num_threads}")
            th.set_num_threads(self.train_args.num_threads)

        if self.train_args.trained_agent != "":
            assert self.train_args.trained_agent.endswith(".zip") and os.path.isfile(
                self.train_args.trained_agent
            ), "The trained_agent must be a valid path to a .zip file"

        print("=" * 10, env_id, " | ", self.train_args.algo, "=" * 10)
        print(f"Seed: {self.train_args.seed}")

        if self.train_args.track:
            try:
                import wandb
            except ImportError as e:
                raise ImportError(
                    "if you want to use Weights & Biases to track experiment, please install W&B via `pip install wandb`"
                ) from e

            run_name = f"{self.train_args.env}__{self.train_args.algo}__{int(time.time())}"

            self.train_args.tensorboard_log = f"{self.train_args.tensorboard_log}/{run_name}"

            wandb.tensorboard.patch(
                root_logdir=self.train_args.tensorboard_log)

            run = wandb.init(name=run_name,
                             project=self.train_args.wandb_project_name,
                             entity=self.train_args.wandb_entity,
                             tags=self.train_args.wandb_tags,
                             config=vars(self.train_args),
                             #  sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                             monitor_gym=True,  # auto-upload the videos of agents playing the game
                             save_code=True)

        self.exp_manager = ExperimentManager(self.train_args,
                                             self.train_args.algo,
                                             env_id,
                                             self.train_args.log_folder,
                                             self.train_args.tensorboard_log,
                                             self.train_args.n_timesteps,
                                             self.train_args.eval_freq,
                                             self.train_args.eval_episodes,
                                             self.train_args.save_freq,
                                             self.train_args.hyperparams,
                                             self.train_args.env_kwargs,
                                             self.train_args.eval_env_kwargs,
                                             self.train_args.trained_agent,
                                             self.train_args.optimize_hyperparameters,
                                             self.train_args.storage,
                                             self.train_args.study_name,
                                             self.train_args.n_trials,
                                             self.train_args.max_total_trials,
                                             self.train_args.n_jobs,
                                             self.train_args.sampler,
                                             self.train_args.pruner,
                                             self.train_args.optimization_log_path,
                                             n_startup_trials=self.train_args.n_startup_trials,
                                             n_evaluations=self.train_args.n_evaluations,
                                             truncate_last_trajectory=self.train_args.truncate_last_trajectory,
                                             uuid_str=uuid_str,
                                             seed=self.train_args.seed,
                                             log_interval=self.train_args.log_interval,
                                             save_replay_buffer=self.train_args.save_replay_buffer,
                                             verbose=self.train_args.verbose,
                                             vec_env_type=self.train_args.vec_env,
                                             n_eval_envs=self.train_args.n_eval_envs,
                                             no_optim_plots=self.train_args.no_optim_plots,
                                             device=self.train_args.device,
                                             config=self.train_args.conf_file,
                                             show_progress=self.train_args.progress)

        # Prepare experiment and launch hyperparameter optimization if needed
        results = self.exp_manager.setup_experiment()

        if results is not None:

            self.model, saved_hyperparams = results

            if self.train_args.track:

                # we need to save the loaded hyperparameters
                self.train_args.saved_hyperparams = saved_hyperparams

                assert run is not None

                run.config.setdefaults(vars(self.train_args))

            # Normal training
            if self.model is not None:
                self.exp_manager.learn(self.model)
                self.exp_manager.save_trained_model(self.model)
        else:
            self.exp_manager.hyperparameters_optimization()
