#!/usr/bin/env python
# coding: utf-8

import os
import time
import sys
import json
import numpy as np
import torch as th
import motornet as mn
from simple_policy import Policy
from simple_task import CentreOutFFMinJerk
from simple_utils import *
from tqdm import tqdm
import pickle
from joblib import Parallel, delayed
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

device = th.device("cpu")

def go(do_distributed):

    effector = mn.effector.RigidTendonArm26(muscle=mn.muscle.RigidTendonHillMuscle())
    env = CentreOutFFMinJerk(effector=effector, max_ep_duration=1.)

    if do_distributed:
        policy = DDP(Policy(env.observation_space.shape[0], 128, env.n_muscles, device=device))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        policy = Policy(env.observation_space.shape[0], 128, env.n_muscles, device=device)
        rank = 0
        world_size = 1

    optimizer = th.optim.Adam(policy.parameters(), lr=10**-3)


    batch_size =    int(256 / world_size)
    n_batch    =   100
    interval   =  1000
    model_name = "simple"

    losses = {
        'overall': [],
        'position': [],
        'angle': [],
        'lateral': [],
        'muscle': [],
        'hidden': []}

    for batch in tqdm(range(n_batch),
                    desc=f"Training {n_batch} batches of {batch_size}",
                    unit="batch"):

        data = run_episode(env, policy, batch_size, catch_trial_perc=50, condition='train', ff_coefficient=0.0, detach=False)
        loss, position_loss, muscle_loss, hidden_loss, angle_loss, lateral_loss = cal_loss(data, env.muscle.max_iso_force, env.dt, policy, test=False)

        # backward pass & update weights
        optimizer.zero_grad() 
        loss.backward()
        th.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.)  # important!
        optimizer.step()

        # save weights/config/losses
        if (batch % interval == 0) and (batch != 0) and rank == 0:
            save_model(env, policy, losses, model_name, quiet=True)
            with open('simple_data.pkl', 'wb') as f:
                pickle.dump(data, f)

        # Update loss values in the dictionary
        losses['overall'].append(loss.item())
        losses['position'].append(position_loss.item())
        losses['angle'].append(angle_loss.item())
        losses['lateral'].append(lateral_loss.item())
        losses['muscle'].append(muscle_loss.item())
        losses['hidden'].append(hidden_loss.item())

    if rank == 0:
        save_model(env, policy, losses, model_name)

    return [env,policy,losses,data]


if __name__ == "__main__":
    
    if (len(sys.argv) < 2):
        print(f"\npython simple.py n, where n=number of models to train\n")
    else:
        do_distributed = int(sys.argv[1])
        if do_distributed == 1:
            dist.init_process_group("gloo")
            go(do_distributed)
            dist.destroy_process_group()
        else:
            go(do_distributed)


