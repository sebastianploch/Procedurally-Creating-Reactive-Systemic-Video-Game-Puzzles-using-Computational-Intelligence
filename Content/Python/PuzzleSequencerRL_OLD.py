import PuzzleSequencerGameState as PSGS
import PuzzleSequencerPuzzles as PSP

# import unreal # Disabled prints?
import math
import os
import random
import sys
import time
import logging
from collections import namedtuple, deque
from itertools import count

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib
import matplotlib.pyplot as plt

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

# check if CUDA is available otherwise run on CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define transitions
transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


# ------------------------------ TEST -----------------------------------------

# game_state = GameState(5, 5)
#
# button = Button([1, 1])
# pressure_plate = PressurePlate([2, 2], button)
# door = Door([3, 3], pressure_plate)
#
# game_state.add_puzzle(button)
# game_state.add_puzzle(pressure_plate)
# game_state.add_puzzle(door)
# game_state.set_terminal_puzzle(door)
# print(game_state.get_map())
#
# action = torch.zeros([len(available_actions)], dtype=torch.float32)
# action[int(available_actions["move_down"])] = 1.  # move down
# game_state.step(action)
#
# action = torch.zeros([len(available_actions)], dtype=torch.float32)
# action[int(available_actions["move_right"])] = 1.  # move down
# game_state.step(action)
#
# action = torch.zeros([len(available_actions)], dtype=torch.float32)
# action[int(available_actions["press"])] = 1.  # move down
# game_state.step(action)
#
# action = torch.zeros([len(available_actions)], dtype=torch.float32)
# action[int(available_actions["move_down"])] = 1.  # move down
# game_state.step(action)
#
# action = torch.zeros([len(available_actions)], dtype=torch.float32)
# action[int(available_actions["move_right"])] = 1.  # move down
# game_state.step(action)
#
# # action = torch.zeros([len(available_actions)], dtype=torch.float32)
# # action[int(available_actions["activate"])] = 1.  # move down
# # game_state.step(action)
#
# action = torch.zeros([len(available_actions)], dtype=torch.float32)
# action[int(available_actions["move_down"])] = 1.  # move down
# game_state.step(action)
#
# action = torch.zeros([len(available_actions)], dtype=torch.float32)
# action[int(available_actions["move_right"])] = 1.  # move down
# game_state.step(action)
#
# action = torch.zeros([len(available_actions)], dtype=torch.float32)
# action[int(available_actions["open"])] = 1.  # move down
# game_state.step(action)


# ----------------------------- NEURAL NETWORK -----------------------------
class DQN(nn.Module):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.num_actions = num_actions
        self.gamma = 0.99
        self.final_epsilon = 0.0001
        self.initial_epsilon = 0.1
        self.num_iterations = 50000
        self.replay_memory_size = 10000
        self.minibatch_size = 32

        self.layer1 = nn.Linear(4, 32, dtype=torch.float32)
        self.relu1 = nn.ReLU(inplace=True)
        self.layer2 = nn.Linear(256, 512, dtype=torch.float32)
        self.relu2 = nn.ReLU(inplace=True)
        self.layer3 = nn.Linear(512, self.num_actions, dtype=torch.float32)

    def forward(self, x):
        out = x.view(x.size()[0], -1)
        out = self.layer1(out)
        out = self.relu1(out)

        out = self.layer2(out)
        out = self.relu2(out)
        out = self.layer3(out)
        return out


# -------------------------------------------------------------------------------------------------------------------
class DQN2(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DQN2, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimiser = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        return actions


class Agent:
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 max_memory_size=100000, eps_end=0.01, eps_dec=5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_memory_size
        self.batch_size = batch_size
        self.mem_cntr = 0

        self.Q_eval = DQN2(self.lr, n_actions=n_actions, input_dims=input_dims,
                           fc1_dims=256, fc2_dims=256)

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            # state = torch.tensor([observation]).to(self.Q_eval.device)
            state = torch.from_numpy(observation).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self, losses):
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimiser.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = torch.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        action_batch = self.action_memory[batch]

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)  # could be turned into target network
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        losses.append(loss.cpu().item())
        loss.backward()
        self.Q_eval.optimiser.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min \
            else self.eps_min


# -------------------------------------------------------------------------------------------------------------------
def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.uniform_(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0.01)


def train(game_state, model, start, losses, q_values, completions):
    optimiser = optim.Adam(model.parameters(), lr=1e-6)
    criterion = nn.MSELoss()
    replay_memory = []

    # Pick action
    action = torch.zeros([model.num_actions], dtype=torch.float32, device=device)
    action[0] = 1

    reward, terminal = game_state.step(action)

    # Create total game state from game state's map and map position
    state = torch.cat((torch.tensor(game_state.get_map(), dtype=torch.float32, device=device),
                       torch.tensor(game_state.get_position_map(), dtype=torch.float32, device=device))).unsqueeze(0)

    epsilon = model.initial_epsilon
    iteration = 0

    epsilon_decrements = np.linspace(model.initial_epsilon, model.final_epsilon, model.num_iterations)

    while iteration < model.num_iterations:
        output = model(state.clone().detach())[0]

        # initialise action
        action = torch.zeros([model.num_actions], dtype=torch.float32, device=device)

        random_action = random.random() <= epsilon
        is_random = False
        if random_action:
            is_random = True
        action_index = [torch.randint(model.num_actions, torch.Size([]), dtype=torch.int32, device=device)
                        if random_action
                        else torch.argmax(output)][0]

        action[action_index] = 1

        reward, terminal = game_state.step(action)

        state_ = torch.cat((torch.tensor(game_state.get_map(), dtype=torch.float32, device=device),
                            torch.tensor(game_state.get_position_map(), dtype=torch.float32, device=device))).unsqueeze(
            0)

        action = action.unsqueeze(0)

        reward = torch.tensor([[reward]], dtype=torch.float32, device=device)

        replay_memory.append((state, action, reward, state_, terminal))

        if len(replay_memory) > model.replay_memory_size:
            replay_memory.pop(0)

        epsilon = epsilon_decrements[iteration]  # update the epsilon

        minibatch = random.sample(replay_memory, min(len(replay_memory), model.minibatch_size))

        state_batch = torch.cat(tuple(d[0] for d in minibatch))
        action_batch = torch.cat(tuple(d[1] for d in minibatch))
        reward_batch = torch.cat(tuple(d[2] for d in minibatch))
        state__batch = torch.cat(tuple(d[3] for d in minibatch))

        output__batch = model(state__batch)  # get output for the next state

        y_batch = torch.cat(tuple(reward_batch[i] if minibatch[i][4]  # terminal state
                                  else reward_batch[i] + model.gamma * torch.max(output__batch[i])
                                  for i in range(len(minibatch))))

        q_value = torch.sum(model(state_batch) * action_batch, dim=1)  # extract the Q-value
        # q_value = model(state_batch).gather(1, action_batch)  # extract the Q-value
        q_values.append(q_value[-1].cpu().item())

        optimiser.zero_grad()  # reset the gradients

        y_batch = y_batch.detach()

        loss = criterion(q_value, y_batch)
        losses.append(loss.cpu().item())

        # backward pass to update the network
        loss.backward()
        optimiser.step()

        state = state_  # update the state for the next step
        iteration += 1

        # TODO: Uncomment once want saving
        if iteration % 5000 == 0:
            torch.save(model, "pretrained_model/current_model_" + str(iteration) + ".pth")

        output_msg = f"Iteration:{iteration} \
        Elapsed Time:{time.time() - start:.5f} \
        Epsilon:{epsilon:.5} \
        Action:{action_index.cpu().detach().numpy()} \
        Reward:{reward.cpu().detach().numpy()[0][0]:.5f} \
        Q Max:{np.max(output.cpu().detach().numpy())} \
        Random:{is_random}"
        logging.info(output_msg)

        if terminal:
            torch.save(model, "pretrained_model/current_model_test" + str(iteration) + ".pth")

            completions.append(iteration)

            # re-initialise the map!
            game_state = initialise_game_state()

            # get state
            state = torch.cat((torch.tensor(game_state.get_map(), dtype=torch.float32, device=device),
                               torch.tensor(game_state.get_position_map(), dtype=torch.float32,
                                            device=device))).unsqueeze(0)


def test(model):
    game_state = initialise_game_state()
    print(game_state.get_map())

    action = torch.zeros([model.num_actions], dtype=torch.float32, device=device)
    action[0] = 1

    reward, terminal = game_state.step(action)
    state = torch.cat((torch.tensor(game_state.get_map(), dtype=torch.float32, device=device),
                       torch.tensor(game_state.get_position_map(), dtype=torch.float32, device=device))).unsqueeze(0)

    while True:
        output = model(state)[0]
        print(f"Output:{output.cpu().detach().numpy()}")

        action = torch.zeros([model.num_actions], dtype=torch.float32, device=device)

        # get action
        action_index = torch.argmax(output).to(device)
        action[action_index] = 1

        # get next state
        reward, terminal = game_state.step(action)
        state_ = torch.cat((torch.tensor(game_state.get_map(), dtype=torch.float32, device=device),
                            torch.tensor(game_state.get_position_map(), dtype=torch.float32, device=device))).unsqueeze(
            0)

        state = state_

        print(f"Action:{action.cpu().detach().numpy()}")

        if terminal:
            print("Finished Game!")


def initialise_game_state():
    game_state = PSGS.GameState(4, 4)

    # Init puzzles
    button = PSP.Button([1, 1])
    # pressure_plate = PSP.PressurePlate([2, 2], button)
    door = PSP.Door([3, 3], button)

    # Set terminal puzzle
    game_state.set_terminal_puzzle(door)

    # Add puzzles to game state
    game_state.add_puzzle(button)
    # game_state.add_puzzle(pressure_plate)
    game_state.add_puzzle(door)

    # print("Available actions for the game state:")
    # print(PSGS.GameState.available_actions)
    # print("\n")
    return game_state


def plot(losses, q_values, completions):
    # Loss
    fig, ax = plt.subplots()
    iterations = np.arange(0, len(losses)).tolist()
    ax.plot(iterations, losses)
    ax.set_title(f"Loss over iterations")
    ax.set_xlabel(f"iterations [COMPLETIONS: {len(completions)}]")
    ax.set_ylabel("loss")

    # Mark completions
    for completion in completions:
        # ax.plot(completion, losses[completion], 'o', color='red')
        ax.axvspan(completion, completion, color='red', alpha=0.5)

    # Q Values
    fig2, ax2 = plt.subplots()
    ax2.plot(iterations, q_values)
    ax2.set_title("Q values over iterations")
    ax2.set_xlabel(f"iterations [COMPLETIONS: {len(completions)}]")
    ax2.set_ylabel("Q values")

    # Mark completions
    for completion in completions:
        ax2.axvspan(completion, completion, color='red', alpha=0.5)

    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())
    else:
        plt.show(block=True)


def main(mode):
    logging.basicConfig(filename='output.txt', level=logging.INFO, format='', filemode='w')
    cuda_is_available = torch.cuda.is_available()

    game_state = initialise_game_state()
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=len(game_state.available_actions),
                  eps_end=0.01, input_dims=[32], lr=0.003)

    scores, eps_history, losses, avg_losses = [], [], [], []
    n_games = 100

    for i in range(n_games):
        score = 0
        done = False
        observation = np.array([game_state.get_map(), game_state.get_position_map()], dtype=np.float32).ravel()

        while not done:
            action = agent.choose_action(observation)
            reward, done = game_state.step(action)
            observation_ = np.array([game_state.get_map(), game_state.get_position_map()], dtype=np.float32).ravel()

            score += reward
            agent.store_transition(observation, action, reward, observation_, done)

            agent.learn(losses)
            observation = observation_

        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_losses.append(np.mean(losses))
        losses.clear()

        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score %.2f' % score,
              'average score %.2f' % avg_score,
              'epsilon %.2f' % agent.epsilon)

        # Reset Game State
        game_state = initialise_game_state()

    x = [i + 1 for i in range(n_games)]
    filename = "test.png"
    plot_learning(x, scores, eps_history, losses, filename)
    plot_learning_loss(x, scores, avg_losses, "test_losses.png")


def plot_learning(x, scores, epsilons, losses, filename, lines=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Game", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t - 20):(t + 1)])

    ax2.scatter(x, running_avg, color="C1")
    # ax2.xaxis.tick_top()
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    # ax2.set_xlabel('x label 2', color="C1")
    ax2.set_ylabel('Score', color="C1")
    # ax2.xaxis.set_label_position('top')
    ax2.yaxis.set_label_position('right')
    # ax2.tick_params(axis='x', colors="C1")
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)


def plot_learning_loss(x, scores, losses, filename, lines=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, losses, color="C0")
    ax.set_xlabel("Game", color="C0")
    ax.set_ylabel("Loss", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t - 20):(t + 1)])

    ax2.scatter(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)


if __name__ == "__main__":
    main(sys.argv[1])
