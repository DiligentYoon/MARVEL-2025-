"""
Main driver script for training MARVEL.

This script initializes and manages a distributed training process using Ray, 
implementing a Soft Actor-Critic (SAC) algorithm with multiple meta agents. 
Key functionalities include:
- Initializing neural networks for policy and Q-value estimation
- Setting up distributed training with multiple workers
- Managing experience replay buffer
- Performing training iterations 
- Logging metrics and saving model checkpoints

The main training loop handles:
- Collecting experiences from distributed workers
- Sampling and training on batches of experiences
- Updating policy, Q-networks, and temperature parameter
- Periodic model checkpointing and performance logging

Supports GPU/CPU training, WandB logging, and model resumption.
"""
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import ray
import os
import numpy as np
import random
import wandb

from utils.model import PolicyNet, QNet
from utils.runner import RLRunner
from parameter import *

ray.init(locals=True)
print("Welcome to MARVEL!")

writer = SummaryWriter(train_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)

if not os.path.exists(load_path):
    os.makedirs(load_path)


def main():
    # use GPU/CPU for driver/worker
    device = torch.device('cuda') if USE_GPU_GLOBAL else torch.device('cpu')
    local_device = torch.device('cuda') if USE_GPU else torch.device('cpu')

    device = torch.device("cpu")
    local_device = torch.device("cpu")

    # initialize neural networks
    global_policy_net = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM, NUM_ANGLES_BIN).to(device)
    global_q_net1 = QNet(NODE_INPUT_DIM, EMBEDDING_DIM, NUM_ANGLES_BIN, TRAIN_ALGO).to(device)
    global_q_net2 = QNet(NODE_INPUT_DIM, EMBEDDING_DIM, NUM_ANGLES_BIN, TRAIN_ALGO).to(device)
    log_alpha = torch.FloatTensor([-2]).to(device)
    log_alpha.requires_grad = True

    global_target_q_net1 = QNet(NODE_INPUT_DIM, EMBEDDING_DIM, NUM_ANGLES_BIN, TRAIN_ALGO).to(device)
    global_target_q_net2 = QNet(NODE_INPUT_DIM, EMBEDDING_DIM, NUM_ANGLES_BIN, TRAIN_ALGO).to(device)

    # initialize optimizers
    global_policy_optimizer = optim.Adam(global_policy_net.parameters(), lr=LR)
    global_q_net1_optimizer = optim.Adam(global_q_net1.parameters(), lr=LR)
    global_q_net2_optimizer = optim.Adam(global_q_net2.parameters(), lr=LR)
    log_alpha_optimizer = optim.Adam([log_alpha], lr=1e-4)

    # target entropy for SAC
    entropy_target = 0.05 * (-np.log(1 / K_SIZE))

    curr_episode = 0
    target_q_update_counter = 1

    if USE_WANDB:
        import parameter
        vars(parameter).__delitem__('__builtins__')
        wandb.init(project='Directional', name=FOLDER_NAME, config=vars(parameter), resume='allow',
                   id=None, notes=None)
        wandb.watch([global_policy_net, global_q_net1], log='all', log_freq=1000, log_graph=False)


    # load model and optimizer trained before
    if LOAD_MODEL:
        print('Loading Model...')
        checkpoint = torch.load(model_path + '/checkpoint.pth', map_location=device)
        global_policy_net.load_state_dict(checkpoint['policy_model'])
        global_q_net1.load_state_dict(checkpoint['q_net1_model'])
        global_q_net2.load_state_dict(checkpoint['q_net2_model'])
        log_alpha = checkpoint['log_alpha']
        log_alpha_optimizer = optim.Adam([log_alpha], lr=1e-4)
        
        global_policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        global_q_net1_optimizer.load_state_dict(checkpoint['q_net1_optimizer'])
        global_q_net2_optimizer.load_state_dict(checkpoint['q_net2_optimizer'])
        log_alpha_optimizer.load_state_dict(checkpoint['log_alpha_optimizer'])
        curr_episode = checkpoint['episode']

    global_target_q_net1.load_state_dict(global_q_net1.state_dict())
    global_target_q_net2.load_state_dict(global_q_net2.state_dict())
    global_target_q_net1.eval()
    global_target_q_net2.eval()

    # launch meta agents
    meta_agents = [RLRunner.remote(i) for i in range(NUM_META_AGENT)]

    # get global networks weights
    weights_set = []
    if device != local_device:
        policy_weights = global_policy_net.to(local_device).state_dict()
        global_policy_net.to(device)
    else:
        policy_weights = global_policy_net.to(local_device).state_dict()
    weights_set.append(policy_weights)

    # distributed training if multiple GPUs available
    dp_policy = nn.DataParallel(global_policy_net)
    dp_q_net1 = nn.DataParallel(global_q_net1)
    dp_q_net2 = nn.DataParallel(global_q_net2)
    dp_target_q_net1 = nn.DataParallel(global_target_q_net1)
    dp_target_q_net2 = nn.DataParallel(global_target_q_net2)

    # launch the first job on each runner
    job_list = []
    for i, meta_agent in enumerate(meta_agents):
        curr_episode += 1
        job_list.append(meta_agent.job.remote(weights_set, curr_episode))

    # initialize metric collector
    metric_name = ['travel_dist', 'success_rate', 'explored_rate']
    training_data = []
    perf_metrics = {}
    for n in metric_name:
        perf_metrics[n] = []

    # initialize training replay buffer
    experience_buffer = []
    for i in range(NUM_EPISODE_BUFFER):
        experience_buffer.append([])

    # collect data from worker and do training
    try:
        while True:
            # wait for any job to be completed
            done_id, job_list = ray.wait(job_list)
            # get the results
            done_jobs = ray.get(done_id)

            # save experience and metric
            for job in done_jobs:
                job_results, metrics, info = job
                for i in range(len(experience_buffer)):
                    experience_buffer[i] += job_results[i]
                for n in metric_name:
                    perf_metrics[n].append(metrics[n])

            # launch new task
            curr_episode += 1
            job_list.append(meta_agents[info['id']].job.remote(weights_set, curr_episode))

            # start training
            if curr_episode % 1 == 0 and len(experience_buffer[0]) >= MINIMUM_BUFFER_SIZE:
                print("Training model!")

                # keep the replay buffer size
                if len(experience_buffer[0]) >= REPLAY_SIZE:
                    for i in range(len(experience_buffer)):
                        experience_buffer[i] = experience_buffer[i][-REPLAY_SIZE:]

                indices = range(len(experience_buffer[0]))

                # training for n times each step
                for j in range(4):
                    # randomly sample a batch data
                    sample_indices = random.sample(indices, BATCH_SIZE)
                    rollouts = []
                    for i in range(len(experience_buffer)):
                        rollouts.append([experience_buffer[i][index] for index in sample_indices])

                    # stack batch data to tensors
                    node_inputs = torch.stack(rollouts[0]).to(device)
                    node_padding_mask = torch.stack(rollouts[1]).to(device)
                    local_edge_mask = torch.stack(rollouts[2]).to(device)
                    current_local_index = torch.stack(rollouts[3]).to(device)
                    current_local_edge = torch.stack(rollouts[4]).to(device)
                    local_edge_padding_mask = torch.stack(rollouts[5]).to(device)
                    frontier_distribution = torch.stack(rollouts[6]).to(device)
                    heading_visited = torch.stack(rollouts[7]).to(device)
                    action = torch.stack(rollouts[8]).to(device)
                    reward = torch.stack(rollouts[9]).to(device)
                    done = torch.stack(rollouts[10]).to(device)
                    next_node_inputs = torch.stack(rollouts[11]).to(device)
                    next_node_padding_mask = torch.stack(rollouts[12]).to(device)
                    next_local_edge_mask = torch.stack(rollouts[13]).to(device)
                    next_current_local_index = torch.stack(rollouts[14]).to(device)
                    next_current_local_edge = torch.stack(rollouts[15]).to(device)
                    next_local_edge_padding_mask = torch.stack(rollouts[16]).to(device)
                    next_frontier_distribution = torch.stack(rollouts[17]).to(device)
                    next_heading_visited = torch.stack(rollouts[18]).to(device)
                    neighbor_best_headings = torch.stack(rollouts[38]).to(device)
                    next_neighbor_best_headings = torch.stack(rollouts[39]).to(device)

                    if TRAIN_ALGO in (2,3):
                        gt_node_inputs = torch.stack(rollouts[19]).to(device)
                        gt_node_padding_mask = torch.stack(rollouts[20]).to(device)
                        gt_edge_mask = torch.stack(rollouts[21]).to(device)
                        gt_current_index = torch.stack(rollouts[22]).to(device)
                        gt_current_edge = torch.stack(rollouts[23]).to(device)
                        gt_edge_padding_mask = torch.stack(rollouts[24]).to(device)
                        gt_frontier_distribution = torch.stack(rollouts[25]).to(device)
                        gt_heading_visited = torch.stack(rollouts[26]).to(device)
                        gt_next_node_inputs = torch.stack(rollouts[27]).to(device)
                        gt_next_node_padding_mask = torch.stack(rollouts[28]).to(device)
                        gt_next_edge_mask = torch.stack(rollouts[29]).to(device)
                        gt_next_current_index = torch.stack(rollouts[30]).to(device)
                        gt_next_current_edge = torch.stack(rollouts[31]).to(device)
                        gt_next_edge_padding_mask = torch.stack(rollouts[32]).to(device)
                        gt_next_frontier_distribution = torch.stack(rollouts[33]).to(device) 
                        gt_next_heading_visited = torch.stack(rollouts[34]).to(device)        

                    if TRAIN_ALGO in (1,3):
                        all_agent_indices = torch.stack(rollouts[35]).to(device)
                        all_agent_next_indices = torch.stack(rollouts[36]).to(device)
                        next_all_agent_next_indices = torch.stack(rollouts[37]).to(device)

                    observation = [node_inputs, node_padding_mask, local_edge_mask, current_local_index,
                                   current_local_edge, local_edge_padding_mask, frontier_distribution, heading_visited, neighbor_best_headings]
                    next_observation = [next_node_inputs, next_node_padding_mask, next_local_edge_mask,
                                        next_current_local_index, next_current_local_edge, next_local_edge_padding_mask, next_frontier_distribution, next_heading_visited, next_neighbor_best_headings]
                    
                    if TRAIN_ALGO == 0:
                        state = observation
                        next_state = next_observation
                    elif TRAIN_ALGO == 1:
                        state = [*observation, all_agent_indices, all_agent_next_indices]
                        next_state = [*next_observation, all_agent_next_indices, next_all_agent_next_indices]
                    elif TRAIN_ALGO == 2:
                        state = [gt_node_inputs, gt_node_padding_mask, gt_edge_mask, gt_current_index,
                                 gt_current_edge, gt_edge_padding_mask, gt_frontier_distribution, gt_heading_visited, neighbor_best_headings]
                        next_state = [gt_next_node_inputs, gt_next_node_padding_mask, gt_next_edge_mask,
                                      gt_next_current_index, gt_next_current_edge, gt_next_edge_padding_mask, gt_next_frontier_distribution, gt_next_heading_visited, next_neighbor_best_headings]
                    elif TRAIN_ALGO == 3:
                        state = [gt_node_inputs, gt_node_padding_mask, gt_edge_mask, gt_current_index,
                                 gt_current_edge, gt_edge_padding_mask, gt_frontier_distribution, gt_heading_visited, neighbor_best_headings, all_agent_indices, all_agent_next_indices]
                        next_state = [gt_next_node_inputs, gt_next_node_padding_mask, gt_next_edge_mask,
                                      gt_next_current_index, gt_next_current_edge, gt_next_edge_padding_mask, gt_next_frontier_distribution, gt_next_heading_visited, next_neighbor_best_headings,
                                      all_agent_next_indices, next_all_agent_next_indices]

                    # SAC
                    with torch.no_grad():
                        q_values1 = dp_q_net1(*state)
                        q_values2 = dp_q_net2(*state)
                        q_values = torch.min(q_values1, q_values2)

                    logp = dp_policy(*observation)
                    policy_loss = torch.sum(
                        (logp.exp().unsqueeze(2) * (log_alpha.exp().detach() * logp.unsqueeze(2) - q_values.detach())),
                        dim=1).mean()

                    global_policy_optimizer.zero_grad()
                    policy_loss.backward()
                    policy_grad_norm = torch.nn.utils.clip_grad_norm_(global_policy_net.parameters(), max_norm=100,
                                                                      norm_type=2)
                    global_policy_optimizer.step()

                    with torch.no_grad():
                        next_logp = dp_policy(*next_observation)
                        next_q_values1 = dp_target_q_net1(*next_state)
                        next_q_values2 = dp_target_q_net2(*next_state)
                        next_q_values = torch.min(next_q_values1, next_q_values2)
                        value_prime = torch.sum(
                            next_logp.unsqueeze(2).exp() * (next_q_values - log_alpha.exp() * next_logp.unsqueeze(2)),
                            dim=1).unsqueeze(1)
                        target_q_batch = reward + GAMMA * (1 - done) * value_prime

                    mse_loss = nn.MSELoss()

                    q_values1 = dp_q_net1(*state)
                    q1 = torch.gather(q_values1, 1, action)
                    q1_loss = mse_loss(q1, target_q_batch.detach()).mean()
                    global_q_net1_optimizer.zero_grad()
                    q1_loss.backward()
                    q_grad_norm = torch.nn.utils.clip_grad_norm_(global_q_net1.parameters(), max_norm=20000,
                                                                 norm_type=2)
                    global_q_net1_optimizer.step()

                    q_values2 = dp_q_net2(*state)
                    q2 = torch.gather(q_values2, 1, action)
                    q2_loss = mse_loss(q2, target_q_batch.detach()).mean()
                    global_q_net2_optimizer.zero_grad()
                    q2_loss.backward()
                    q_grad_norm = torch.nn.utils.clip_grad_norm_(global_q_net2.parameters(), max_norm=20000,
                                                                 norm_type=2)
                    global_q_net2_optimizer.step()

                    entropy = (logp * logp.exp()).sum(dim=-1)
                    alpha_loss = -(log_alpha * (entropy.detach() + entropy_target)).mean()

                    log_alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    log_alpha_optimizer.step()

                    target_q_update_counter += 1

                # data record to be written in tensorboard
                perf_data = []
                for n in metric_name:
                    perf_data.append(np.nanmean(perf_metrics[n]))
                data = [reward.mean().item(), value_prime.mean().item(), policy_loss.item(), q1_loss.item(),
                        entropy.mean().item(), policy_grad_norm.item(), q_grad_norm.item(), log_alpha.item(),
                        alpha_loss.item(), *perf_data]
                training_data.append(data)

            # write record to tensorboard
            if len(training_data) >= SUMMARY_WINDOW:
                write_to_tensor_board(writer, training_data, curr_episode)
                training_data = []
                perf_metrics = {}
                for n in metric_name:
                    perf_metrics[n] = []

            # get the updated global weights
            weights_set = []
            if device != local_device:
                policy_weights = global_policy_net.to(local_device).state_dict()
                global_policy_net.to(device)
            else:
                policy_weights = global_policy_net.to(local_device).state_dict()
            weights_set.append(policy_weights)

            # update the target q net
            if target_q_update_counter > 64:
                print("Updating target q net")
                target_q_update_counter = 1
                global_target_q_net1.load_state_dict(global_q_net1.state_dict())
                global_target_q_net2.load_state_dict(global_q_net2.state_dict())
                global_target_q_net1.eval()
                global_target_q_net2.eval()

            # save the model
            if curr_episode % 32 == 0:
                print('Saving model', end='\n')
                checkpoint = {"policy_model": global_policy_net.state_dict(),
                              "q_net1_model": global_q_net1.state_dict(),
                              "q_net2_model": global_q_net2.state_dict(),
                              "log_alpha": log_alpha,
                              "policy_optimizer": global_policy_optimizer.state_dict(),
                              "q_net1_optimizer": global_q_net1_optimizer.state_dict(),
                              "q_net2_optimizer": global_q_net2_optimizer.state_dict(),
                              "log_alpha_optimizer": log_alpha_optimizer.state_dict(),
                              "episode": curr_episode,
                              }
                path_checkpoint = "./" + model_path + "/checkpoint.pth"
                torch.save(checkpoint, path_checkpoint)
                print('Saved model', end='\n')

    except KeyboardInterrupt:
        print("CTRL_C pressed. Killing remote workers")
        for a in meta_agents:
            ray.kill(a)
        ray.shutdown()
        if USE_WANDB:
            wandb.finish(quiet=True)

def write_to_tensor_board(writer, tensorboard_data, curr_episode):
    tensorboard_data = np.array(tensorboard_data)
    tensorboard_data = list(np.nanmean(tensorboard_data, axis=0))
    reward, value, policy_loss, q_value_loss, entropy, policy_grad_norm, q_value_grad_norm, log_alpha, alpha_loss, travel_dist, success_rate, explored_rate = tensorboard_data

    writer.add_scalar(tag='Losses/Value', scalar_value=value, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Policy Loss', scalar_value=policy_loss, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Alpha Loss', scalar_value=alpha_loss, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Q Value Loss', scalar_value=q_value_loss, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Entropy', scalar_value=entropy, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Policy Grad Norm', scalar_value=policy_grad_norm, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Q Value Grad Norm', scalar_value=q_value_grad_norm, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Log Alpha', scalar_value=log_alpha, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Reward', scalar_value=reward, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Travel Distance', scalar_value=travel_dist, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Explored Rate', scalar_value=explored_rate, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Success Rate', scalar_value=success_rate, global_step=curr_episode)

if __name__ == "__main__":
    main()
