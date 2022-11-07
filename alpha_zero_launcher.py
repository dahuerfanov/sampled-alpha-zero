
from typing import Dict
import random
import numpy as np
import threading
import time
import torch
import os

from agent import Agent
from data_handler import DataProvider, save_new_samples
from game import game_reward, step, reflect, stateToInt
from nnet.model import NNet
from plotter import Plotter


best_model_name_lock = threading.Lock()
best_model_name = None


# episode simulation of agent1 vs agent2:
def simulate_episode(agent1: Agent, agent2: Agent, args: Dict):
    s = torch.zeros(2, args.rows, args.cols, dtype=torch.float32, device=torch.device("cpu"))
    agent = agent1
    flag = 1

    while True:
        for _ in range(args.num_mcts_sims):
            agent.mcts.search(s, agent.nnet)

        best_a, max_pi = -1, -1e10
        pi = agent.mcts.pi(s, tau=0)
        for a in range(args.cols):
            if s[0][0][a] + s[1][0][a] == 0:
                if max_pi < pi[a]:
                    best_a = a
                    max_pi = pi[a]
        a = best_a

        step(s, a, 0, args)
        r, done = game_reward(s, 0, args)
        if done:
            if flag == 1:
                if r > 0:
                    return 1
                else:
                    return 0
            else:
                if r > 0:
                    return -1
                else:
                    return 0

        if flag == 1:
            agent = agent2
        else:
            agent = agent1

        s = torch.flip(s, [0])
        flag = 1 - flag


def evaluate_model(models_path: str,
                   current_model_name: str,
                   device: str,
                   args: Dict) -> None:

    global best_model_name

    with best_model_name_lock:
        if best_model_name is None:
            best_model_name = current_model_name
            print("new model selected!")
            return

        agent1 = Agent(device, os.path.join(models_path, best_model_name), args)
        agent2 = Agent(device, os.path.join(models_path, current_model_name), args)

    agent1.nnet.eval()
    agent2.nnet.eval()
    print("calculating pit rate...")
    count = 0
    tot = 0
    for _ in range(args.num_eps_pit // 2):
        sg = simulate_episode(agent1, agent2, args)
        count += max(sg, 0)
        if sg != 0:
            tot += 1

    for _ in range(args.num_eps_pit // 2, args.num_eps_pit):
        sg = simulate_episode(agent2, agent1, args)
        count += max(-sg, 0)
        if sg != 0:
            tot += 1

    if tot == 0:
        rate, rounds = 0.5, 0
    else:
        rate, rounds = count / tot, tot

    print("rate nnet2 vs nnet1: ", rate, n)
    if rate >= args.threshold:
        with best_model_name_lock:
            best_model_name = current_model_name
        print("better model found!:", current_model_name)


def self_play_episode(agent, args):
    agent.nnet.eval()
    agent.mcts.clear()

    samples_s = []
    samples_v = []

    s = torch.zeros((2, args.rows, args.cols), dtype=torch.float32, device=torch.device("cpu"))
    step_count = 0

    while True:
        for _ in range(args.num_mcts_sims):
            agent.mcts.search(s, agent.nnet)
        samples_s.append(s.clone())

        step_count += 1
        tau = int(step_count <= args.n_threshold_exp)

        pi = agent.mcts.pi(s, tau=tau).numpy()
        a = np.random.choice(args.cols, p=pi)

        if s[0][0][a] + s[1][0][a] != 0:
            a = random.choice([col for col in range(args.cols) if s[0][0][col] + s[1][0][col] == 0])
        step(s, a, 0, args)
        r, done = game_reward(s, 0, args)
        if done:
            for _ in range(len(samples_s)):
                samples_v.append(torch.tensor(r, requires_grad=False, device=torch.device("cpu")))
                r = r * (-1)
            samples_v = samples_v[::-1]

            last = len(samples_s)
            for i in range(last):
                samples_s.append(reflect(samples_s[i]))
            samples_dist = [agent.mcts.pi(s_i) for s_i in samples_s]
            samples_v = samples_v * 2

            return samples_s, samples_dist, samples_v
        else:
            s = torch.flip(s, [0])


def self_play(data_path: str, models_path: str, device: str, args: Dict):

    global best_model_name

    while True:
        if not best_model_name is None:
            agent = Agent(device, os.path.join(models_path, best_model_name), args)
        else:
            agent = Agent(device, None, args)
        print(agent.nnet)

        for e in range(args.num_eps):
            s1, s2, s3 = self_play_episode(agent, args)
            save_new_samples(data_path, s1, s2, s3)


def policy_iteration(data_path: str,
                     models_path: str,
                     model_name: str,
                     figures_path: str,
                     device: str,
                     args: Dict):

    random.seed(0)
    best_model_name = model_name

    thread_self_play = threading.Thread(target=self_play,
                                        args=(data_path, models_path, device, args))
    thread_self_play.start()
    time.sleep(args.sleep_secs_before_train)
   
    data_prov = DataProvider(data_path, args.max_num_samples_mem)    
    plotter = Plotter(figures_path)

    for it in range(args.num_iters):
        current_model_name = f"model_it{it}"
        nnet = NNet("NNet", device, args)
        X, Y_val, Y_distr = data_prov.select_samples(args.sample_size)

        stats = nnet.run_training(X, Y_val, Y_distr)
        torch.save(nnet.state_dict(), os.path.join(models_path, current_model_name))
        plotter.save_plots(it,
                           stats['p_train_losses'],
                           stats['p_val_losses'],
                           stats['v_train_losses'],
                           stats['v_val_losses'],
                           stats['v_train_acc'],
                           stats['v_val_acc'])

        thread_evaluator = threading.Thread(target=evaluate_model,
                                            args=(models_path, current_model_name, device, args))
        thread_evaluator.start()
