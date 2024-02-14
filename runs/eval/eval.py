import os
import sys

import importlib
import numpy as np
import pandas as pd
from pathlib import Path

MODULE_NAME = "infer"
MAIN_PATH = "/home/sequenzia/dev/repos/atari-rl"

PROJECT = "solen-rl-project-eval-2"

NO_RENDER = True

N_ENVS = 2
N_STEPS = 10000

module_path = f"{MAIN_PATH}/utils/{MODULE_NAME}.py"
agents_path = f"{MAIN_PATH}/agents"
data_path = f"{MAIN_PATH}/data"

spec = importlib.util.spec_from_file_location(MODULE_NAME, module_path)
infer = importlib.util.module_from_spec(spec)
sys.modules[MODULE_NAME] = infer
spec.loader.exec_module(infer)


all_infer_logs = {}
all_infer_data = {}


ALGOS = ["ppo", "a2c"]

GAMES = ["Breakout",
         "Pong",
         "SpaceInvaders"]
        #  "Qbert"]
        #  "Seaquest",
        #  "Centipede",
        #  "MsPacman",
        #  "Asterix",
        #  "Asteroids",
        #  "Assault"]

for algo in ALGOS:

    for game in GAMES:
        
        ENV_ID = f"ALE/{game}-v5"

        RUN_KEY = f"{algo.upper()}_{game}"

        infer_logs = infer.infer(run_key=RUN_KEY,
                                 env_id=ENV_ID,
                                 algo=algo,
                                 game=game,
                                 agents_path=agents_path,
                                 n_envs=N_ENVS,
                                 n_steps=N_STEPS,
                                 no_render=NO_RENDER,
                                 project=PROJECT,
                                 debug_on=False)
        
        # all_infer_logs[RUN_KEY] = infer_logs
        
        infer_data_np = np.empty((0,5))

        for idx in range(len(infer_logs)):

            infer_data_np = np.vstack((infer_data_np, 
                                       np.array([infer_logs[idx].scores, 
                                                 infer_logs[idx].times, 
                                                 infer_logs[idx].lengths,
                                                 infer_logs[idx].frame_numbers,
                                                 infer_logs[idx].run_frame_numbers]).T))

        infer_data = pd.DataFrame(infer_data_np, 
                                  columns=["scores", 
                                           "times", 
                                           "lengths", 
                                           "frame_numbers", 
                                           "run_frame_numbers"])

        all_infer_data[RUN_KEY] = infer_data
