# import unreal # Disabled prints?
import math
import os
import random
import sys
import time
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transitions
transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

# ----------------------------- LOGIC -----------------------------
# Test Sequence = Press Button -> Open Door
available_actions = {"idle": 0., "press": 1., "open": 2.}
"""All available actions in the world"""


class PuzzleObject:
    global available_actions

    def __init__(self, position, available_states):
        self.current_state = 0.
        self.position = position
        self.available_states = available_states
        self.completed = False
        self.depends_on = None

    def get_current_state(self):
        return self.current_state

    def get_position(self):
        return self.position

    def get_available_states(self):
        return self.available_states

    def is_completed(self):
        return self.completed

    def get_depends_on(self):
        return self.depends_on

    # To be filled by derived class
    def update_puzzle(self, action):
        pass


class Button(PuzzleObject):
    def __init__(self, position):
        super().__init__(position, {"unpressed": 0., "pressed": 1.})

    def update_puzzle(self, action):
        if ((action == 1.).nonzero(as_tuple=True)[0]) == available_actions["press"]:  # checking the one hot encoding
            if self.completed:
                return False

            self.current_state = self.available_states["pressed"]
            self.completed = True
        return self.completed


class Door(PuzzleObject):
    def __init__(self, position, depends_on):
        super().__init__(position, {"locked": 0., "closed": 1., "open": 2.})
        self.depends_on = depends_on

    def update_puzzle(self, action):
        # Early-out completed puzzle
        if self.completed:
            return False

        # Unlock door if dependant object is completed
        if self.depends_on.is_completed() and self.current_state is self.available_states["locked"]:
            self.current_state = self.available_states["closed"]
            return True

        # Open unlocked door
        if self.current_state == self.available_states["closed"] and \
                ((action == 1.).nonzero(as_tuple=True)[0]) == available_actions["open"]:
            self.current_state = self.available_states["open"]
            self.completed = True
            return True

        return False


class GameState:
    def __init__(self, map_width, map_height):
        self.puzzles = []
        self.terminal_puzzle = None
        self.map_width = map_width
        self.map_height = map_height
        self.map = np.full((map_width, map_height), -1.)

    def step(self, action):
        is_terminal = False
        reward = 0.1

        # Update puzzle pieces with action
        for puzzle in self.puzzles:
            if puzzle.update_puzzle(action):
                reward += 0.25  # Increase reward if puzzle succeeded with provided action
                puzzle_position = puzzle.get_position()
                self.map[puzzle_position[0], puzzle_position[1]] = puzzle.get_current_state()

                # Update dependants
                depends_on = puzzle.get_depends_on()
                if depends_on:
                    puzzle_position = depends_on.get_position()
                    self.map[puzzle_position[0], puzzle_position[1]] = depends_on.get_current_state()

                # Terminate if the terminal puzzle is reached and completed
                if puzzle is self.terminal_puzzle and puzzle.is_completed():
                    is_terminal = True
                    reward = 1.
                    break
            else:
                reward -= 0.15

        print(f"action: {action}, reward: {reward}, terminal: {is_terminal}")
        return reward, is_terminal

    def add_puzzle(self, puzzle):
        puzzle_x = puzzle.get_position()[0]
        puzzle_y = puzzle.get_position()[1]
        self.map[puzzle_x, puzzle_y] = 0.
        self.puzzles.append(puzzle)

    def set_terminal_puzzle(self, puzzle):
        self.terminal_puzzle = puzzle

    def get_map(self):
        return self.map

    def get_map_size(self):
        return self.map_width * self.map_height

    def get_puzzles(self):
        return self.puzzles


# ------------------------------ TEST -----------------------------------------
# button = Button([1, 1])
# door = Door([2, 2], button)
# final_door = Door([4, 4], door)
#
# gs = GameState(5, 5)
# gs.set_terminal_puzzle(final_door)
#
# gs.add_puzzle(button)
# gs.add_puzzle(door)
# gs.add_puzzle(final_door)
# print(gs.get_map())
# print("\n")
#
# a = torch.zeros([len(available_actions)], dtype=torch.float32)
# a[1] = 1
# gs.step(a)
# print(gs.get_map())
# print("\n")
#
# a[1] = 0
# a[2] = 1
# gs.step(a)
# print(gs.get_map())
# print("\n")
#
# gs.step(a)
# print(gs.get_map())
# print("\n")


# class GameStateOld:
#     game_states = {"Idle": 0, "In-Progress": 1, "Finished": 2}
#
#     def __init__(self, puzzles, terminal_puzzle):
#         self.current_state = self.game_states["Idle"]
#         self.puzzles = puzzles
#         self.terminal_puzzle = terminal_puzzle
#
#     def step(self, action):
#         is_terminal = False
#         reward = 0.1
#
#         # Early-out if game state is finished
#         if self.current_state == self.game_states["Finished"]:
#             return -1, True
#
#         # Change state to in-progress if stepping idle state
#         if self.current_state is self.game_states["Idle"]:
#             self.current_state = self.game_states["In-Progress"]
#
#         # Update puzzle pieces with action
#         for puzzle in self.puzzles:
#             if puzzle.update_puzzle(action):
#                 reward += 0.25  # Increase reward if puzzle succeeded with provided action
#
#                 # Terminate if the terminal puzzle is reached and completed
#                 if puzzle is self.terminal_puzzle:
#                     self.current_state = self.game_states["Finished"]
#                     is_terminal = True
#                     reward = 1
#                     break
#             else:
#                 reward -= 0.15
#
#         print(f"action: {action}, reward: {reward}, terminal: {is_terminal}")
#         return reward, is_terminal
#
#     def get_current_state(self):
#         return self.current_state
#
#     def get_puzzles(self):
#         return self.puzzles


# ----------------------------- NEURAL NETWORK -----------------------------
class DQN(nn.Module):

    def __init__(self, num_actions=3):
        super(DQN, self).__init__()
        self.num_actions = num_actions
        self.gamma = 0.99
        self.final_epsilon = 0.0001
        self.initial_epsilon = 0.1
        self.num_iterations = 300
        self.replay_memory_size = 50
        self.minibatch_size = 32

        # self.conv1 = nn.Conv2d(3, 32, 8, 4)
        # self.relu1 = nn.ReLU(inplace=True)
        # self.conv2 = nn.Conv2d(32, 64, 4, 2)
        # self.relu2 = nn.ReLU(inplace=True)
        # self.conv3 = nn.Conv2d(64, 64, 3, 1)
        # self.relu3 = nn.ReLU(inplace=True)
        # self.fc4 = nn.Linear(3136, 512)
        # self.relu4 = nn.ReLU(inplace=True)
        # self.fc5 = nn.Linear(512, self.num_actions)

        self.layer1 = nn.Linear(4, 32, dtype=torch.float64)
        self.relu1 = nn.ReLU(inplace=True)
        self.layer2 = nn.Linear(128, 512, dtype=torch.float64)
        self.relu2 = nn.ReLU(inplace=True)
        self.layer3 = nn.Linear(512, self.num_actions, dtype=torch.float64)

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

        out = self.layer1(x)
        out = self.relu1(out)
        out = out.view(out.size()[0], -1)
        out = self.layer2(out)
        out = self.relu2(out)
        out = self.layer3(out)

        return out


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.uniform(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0.01)


def train(model, start, losses, q_values):
    optimiser = optim.Adam(model.parameters(), lr=1e-6)
    criterion = nn.MSELoss()
    replay_memory = []

    # initialise the map!
    button = Button([1, 1])
    door = Door([2, 2], button)

    game_state = GameState(4, 4)
    game_state.set_terminal_puzzle(door)

    game_state.add_puzzle(button)
    game_state.add_puzzle(door)
    print(game_state.get_map())
    print("\n")

    action = torch.zeros([model.num_actions], dtype=torch.float64)
    action[0] = 1
    if torch.cuda.is_available():
        action = action.cuda()

    reward, terminal = game_state.step(action)
    state = torch.tensor(game_state.get_map()).unsqueeze(0)

    if torch.cuda.is_available():
        state = state.cuda()

    epsilon = model.initial_epsilon
    iteration = 0

    epsilon_decrements = np.linspace(model.initial_epsilon, model.final_epsilon, model.num_iterations)

    while iteration < model.num_iterations:
        output = model(torch.tensor(state).clone().detach())[0]

        action = torch.zeros([model.num_actions], dtype=torch.float64)  # initialise action
        if torch.cuda.is_available():
            action = action.cuda()

        random_action = random.random() <= epsilon
        if random_action:
            print("picked random action :)")
        action_index = [torch.randint(model.num_actions, torch.Size([]), dtype=torch.int32)
                        if random_action
                        else torch.argmax(output)][0]

        if torch.cuda.is_available():
            action_index = action_index.cuda()

        action[action_index] = 1.

        reward, terminal = game_state.step(action)
        # state_ = game_state.get_map()
        state_ = torch.tensor(game_state.get_map()).unsqueeze(0)
        if torch.cuda.is_available():
            state_ = state_.cuda()

        action = action.unsqueeze(0)
        reward = torch.from_numpy(np.array([reward], dtype=np.float64)).unsqueeze(0)

        replay_memory.append((state, action, reward, state_, terminal))

        if len(replay_memory) > model.replay_memory_size:
            replay_memory.pop(0)

        epsilon = epsilon_decrements[iteration]  # update the epsilon

        minibatch = random.sample(replay_memory, min(len(replay_memory), model.minibatch_size))

        state_batch = torch.cat(tuple(d[0] for d in minibatch))
        action_batch = torch.cat(tuple(d[1] for d in minibatch))
        reward_batch = torch.cat(tuple(d[2] for d in minibatch))
        state__batch = torch.cat(tuple(d[3] for d in minibatch))

        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            state__batch = state__batch.cuda()

        output__batch = model(state__batch)  # get output for the next state

        y_batch = torch.cat(tuple(reward_batch[i] if minibatch[i][4]  # terminal state
                                  else reward_batch[i] + model.gamma * torch.max(output__batch[i])
                                  for i in range(len(minibatch))))

        q_value = torch.sum(model(state_batch) * action_batch, dim=1)  # extract the Q-value
        # q_value = model(state_batch).gather(1, action_batch)  # extract the Q-value
        q_values = q_value

        optimiser.zero_grad()  # reset the gradients

        y_batch = y_batch.detach()

        loss = criterion(q_value, y_batch)
        losses.append(loss.cpu().item())

        # backward pass to update the network
        loss.backward()
        optimiser.step()

        state = state_  # update the state for the next step
        iteration += 1

        if iteration % 50 == 0:
            torch.save(model, "pretrained_model/current_model_" + str(iteration) + ".pth")

        print("iteration:", iteration, "elapsed time:", time.time() - start, "epsilon:", epsilon, "action:",
              action_index.cpu().detach().numpy(), "reward:", reward.numpy()[0][0], "Q max:",
              np.max(output.cpu().detach().numpy()))

        if terminal:
            return


def test(model):
    button = Button()
    door = Door(button)
    game_state = GameState([button, door], door)

    # TODO should this be a tensor instead? like in the tutorial
    # action = torch.zeros([model.num_actions], dtype=torch.float64)
    # action[0] = 0
    action = 0
    reward, terminal = game_state.step(action)
    state = game_state.get_map()

    while True:
        output = model(state)[0]

        action = torch.zeros([model.num_actions], dtype=torch.float64)
        if torch.cuda.is_available():
            action = action.cuda()

        action_index = torch.argmax(output)
        if torch.cuda.is_available():
            action_index = action_index.cuda()
        action[action_index] = 0

        # TODO determine next state and reward :D (the cool stuffz)
        reward, terminal = game_state.step(action)
        state_ = game_state.get_map()

        state = state_  # TODO do we need to break at some point? lol the git code doesn't for some reason


def main(mode):
    cuda_is_available = torch.cuda.is_available()

    if mode == 'test':
        model = torch.load(
            'pretrained_model/current_model_300.pth',
            map_location='cpu' if not cuda_is_available else None
        ).eval()

        if cuda_is_available:  # put on GPU if CUDA is available
            model = model.cuda()

        test(model)

    elif mode == 'train':
        if not os.path.exists('pretrained_model/'):
            os.mkdir('pretrained_model/')

        model = DQN(len(available_actions))

        if cuda_is_available:
            model = model.cuda()

        model.apply(init_weights)
        start = time.time()
        losses = []
        q_values = []

        train(model, start, losses, q_values)

        fig, ax = plt.subplots()
        iterations = np.arange(0, len(losses)).tolist()
        ax.plot(iterations, losses, linewidth=2.0)

        # fig2, ax2 = plt.subplots()
        # ax2.plot(iterations, q_values, linewidth=2.0)

        plt.show(block=True)


if __name__ == "__main__":
    main(sys.argv[1])

# # # Cyclic buffer used to hold observed transitions
# class ReplayMemory(object):
#     def __init__(self, capacity):
#         self.memory = deque([], maxlen=capacity)
#
#     def push(self, *args):
#         """Save a Transition"""
#         self.memory.append(Transition(*args))
#
#     def sample(self, batch_size):
#         return random.sample(self.memory, batch_size)
#
#     def __len__(self):
#         return len(self.memory)

# # Q Network class
# class DQN(nn.Module):
#     def __init__(self, h, w, outputs):
#         super(DQN, self).__init__()
#         self.conv1 = nn.Conv2d(4, 16, kernel_size=5, stride=2)
#         self.bn1 = nn.BatchNorm2d(16)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
#         self.bn2 = nn.BatchNorm2d(32)
#         self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
#         self.bn3 = nn.BatchNorm2d(32)
#
#         # Number of Linear input connections depends on output of conv2d layers
#         def conv2d_size_out(size, kernel_size=5, stride=2):
#             return (size - (kernel_size - 1) - 1) // stride + 1
#
#         convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
#         convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
#         linear_input_size = convw * convh * 32
#         self.head = nn.Linear(linear_input_size, outputs)
#
#     def forward(self, x):
#         x = x.to(Device)
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         return self.head(x.view(x.size(), -1))
#
#
# # Training - Hyperparameters
# BATCH_SIZE = 128
# GAMMA = 0.999
# EPSILON_START = 0.9
# EPSILON_END = 0.05
# EPSILON_DECAY = 200
# TARGET_UPDATE = 10

# Training

# # Get number of actions from gym action space
# n_actions = env.action_space.n
#
# policy_net = DQN(screen_height, screen_width, n_actions).to(device)
# target_net = DQN(screen_height, screen_width, n_actions).to(device)
# target_net.load_state_dict(policy_net.state_dict())
# target_net.eval()
#
# optimizer = optim.RMSprop(policy_net.parameters())
#
# steps_done = 0
# memory = ReplayMemory(10000)

# def select_action(state):
#     global steps_done
#     sample = random.random()
#     epsilon_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * \
#                         math.exp(-1. * steps_done / EPSILON_DECAY)
#     steps_done += 1
#     if sample > epsilon_threshold:
#         with torch.no_grad():
#             return  # policy_net(state).max(1)[1].view(1, 1)
#     else:
#         return  # torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
#
#
# episode_durations = []
#
#
# def plot_durations():
#     plt.figure(2)
#     plt.clf()
#     durations_t = torch.tensor(episode_durations, dtype=torch.float)
#     plt.title("Training...")
#     plt.xlabel("Episode")
#     plt.ylabel("Duration")
#     plt.plot(durations_t.numpy())
#
#     # Take 100 episode averages and plot them
#     if len(durations_t) >= 100:
#         means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
#         means = torch.cat((torch.zeros(99), means))
#         plt.plot(means.numpy())
#
#     plt.pause(0.001)
#     if is_ipython:
#         display.clear_output(wait=True)
#         display.display(plt.gcf())
#
#
# # Training Loop
# def optimise_model():
#     if len(memory) < BATCH_SIZE:
#         return
#     transitions = memory.sample(BATCH_SIZE)
#     batch = Transition(*zip(*transitions))
#     non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
#                                   device=Device, dtype=torch.bool)
#     non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
#
#     state_batch = torch.cat(batch.state)
#     action_batch = torch.cat(batch.action)
#     reward_batch = torch.cat(batch.reward)
#
#     # # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
#     # # columns of actions taken. These are the actions which would've been taken
#     # # for each batch state according to policy_net
#     # state_action_values = policy_net(state_batch).gather(1, action_batch)
#     #
#     # # Compute V(s_{t+1}) for all next states.
#     # # Expected values of actions for non_final_next_states are computed based
#     # # on the "older" target_net; selecting their best reward with max(1)[0].
#     # # This is merged based on the mask, such that we'll have either the expected
#     # # state value or 0 in case the state was final.
#     # next_state_values = torch.zeros(BATCH_SIZE, device=device)
#     # next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
#     # # Compute the expected Q values
#     # expected_state_action_values = (next_state_values * GAMMA) + reward_batch
#
#     # # Compute Huber Loss
#     # critetion = nn.SmoothL1Loss()
#     # loss = critetion(state_actions_values, expected_state_action_values.unsqueeze(1))
#
#     # # Optimize the model
#     # optimizer.zero_grad()
#     # loss.backward()
#     # for param in policy_net.parameters():
#     #     param.grad.data.clamp_(-1, 1)
#     # optimizer.step()
#
# # Main Loop
