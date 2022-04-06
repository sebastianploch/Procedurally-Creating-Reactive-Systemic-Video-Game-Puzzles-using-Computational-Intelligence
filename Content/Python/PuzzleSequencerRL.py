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
import torchvision.transforms as T

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

        # self.conv1 = nn.Conv2d(4, 32, 8, 4)
        # self.relu1 = nn.ReLU(inplace=True)
        # self.conv2 = nn.Conv2d(32, 64, 4, 2)
        # self.relu2 = nn.ReLU(inplace=True)
        # self.conv3 = nn.Conv2d(64, 64, 3, 1)
        # self.relu3 = nn.ReLU(inplace=True)
        # self.fc4 = nn.Linear(3136, 512)
        # self.relu4 = nn.ReLU(inplace=True)
        # self.fc5 = nn.Linear(512, self.number_of_actions)

        self.layer1 = nn.Linear(4, 32, dtype=torch.float32)
        self.relu1 = nn.ReLU(inplace=True)
        self.layer2 = nn.Linear(256, 512, dtype=torch.float32)
        self.relu2 = nn.ReLU(inplace=True)
        self.layer3 = nn.Linear(512, self.num_actions, dtype=torch.float32)

    def forward(self, x):
        # out = self.conv1(x)
        # out = self.relu1(out)
        # out = self.conv2(out)
        # out = self.relu2(out)
        # out = self.conv3(out)
        # out = self.relu3(out)
        # out = out.view(out.size()[0], -1)
        # out = self.fc4(out)
        # out = self.relu4(out)
        # out = self.fc5(out)

        out = x.view(x.size()[0], -1)
        out = self.layer1(out)
        out = self.relu1(out)

        out = self.layer2(out)
        out = self.relu2(out)
        out = self.layer3(out)
        return out


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

    print("Available actions for the game state:")
    print(PSGS.GameState.available_actions)
    print("\n")
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

    if mode == 'test':
        model = torch.load(
            'pretrained_model/current_model_test6176.pth',
            map_location='cpu' if not cuda_is_available else None
        ).eval().to(device)

        test(model)

    elif mode == 'train':
        if not os.path.exists('pretrained_model/'):
            os.mkdir('pretrained_model/')

        game_state = initialise_game_state()

        model = DQN(len(PSGS.GameState.available_actions)).to(device)
        model.apply(init_weights)

        start = time.time()
        losses = []
        q_values = []
        completions = []

        train(game_state, model, start, losses, q_values, completions)

        plot(losses, q_values, completions)


if __name__ == "__main__":
    main(sys.argv[1])
