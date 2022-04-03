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
available_actions = {"idle": 0., "press": 1., "open": 2.,  # Puzzle Interaction
                     "select_up": 3., "select_down": 4., "select_left": 5., "select_right": 6.}  # Grid Selection
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
            return True

        return False


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
            # return True

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

        self.current_grid_pos_x = 0
        self.current_grid_pos_y = 0
        self.selected_puzzle = None

    def step(self, action):
        is_terminal = False
        reward = 0.

        # Early-out idle action
        if ((action == 1.).nonzero(as_tuple=True)[0]) == available_actions["idle"]:
            return reward, is_terminal

        # Select grid
        if ((action == 1.).nonzero(as_tuple=True)[0]) == available_actions["select_up"]:
            if self.current_grid_pos_y + 1 < self.map_height:
                reward += 0.01
                self.current_grid_pos_y += 1
                print(f"current grid pos X: {self.current_grid_pos_x} Y: {self.current_grid_pos_y}\n")
                self.update_selected_puzzle()
            else:
                reward -= 0.5
            return reward, is_terminal

        elif ((action == 1.).nonzero(as_tuple=True)[0]) == available_actions["select_down"]:
            if self.current_grid_pos_y - 1 >= 0:
                reward += 0.01
                self.current_grid_pos_y -= 1
                print(f"current grid pos X: {self.current_grid_pos_x} Y: {self.current_grid_pos_y}\n")
                self.update_selected_puzzle()
            else:
                reward -= 0.5
            return reward, is_terminal

        elif ((action == 1.).nonzero(as_tuple=True)[0]) == available_actions["select_left"]:
            if self.current_grid_pos_x - 1 >= 0:
                reward += 0.01
                self.current_grid_pos_x -= 1
                print(f"current grid pos X: {self.current_grid_pos_x} Y: {self.current_grid_pos_y}\n")
                self.update_selected_puzzle()
            else:
                reward -= 0.5
            return reward, is_terminal

        elif ((action == 1.).nonzero(as_tuple=True)[0]) == available_actions["select_right"]:
            if self.current_grid_pos_x + 1 < self.map_width:
                reward += 0.01
                self.current_grid_pos_x += 1
                print(f"current grid pos X: {self.current_grid_pos_x} Y: {self.current_grid_pos_y}\n")
                self.update_selected_puzzle()
            else:
                reward -= 0.5
            return reward, is_terminal

        # Update puzzle it is in the selected grid space
        if self.selected_puzzle is not None:
            if self.selected_puzzle.update_puzzle(action):
                reward += 0.25  # Increase reward if puzzle succeeded with provided action
                puzzle_position = self.selected_puzzle.get_position()
                self.map[puzzle_position[0], puzzle_position[1]] = self.selected_puzzle.get_current_state()

                # Update dependants
                depends_on = self.selected_puzzle.get_depends_on()
                if depends_on:
                    puzzle_position = depends_on.get_position()
                    self.map[puzzle_position[0], puzzle_position[1]] = depends_on.get_current_state()

                # Terminate if the terminal puzzle is reached and completed
                if self.selected_puzzle is self.terminal_puzzle and self.selected_puzzle.is_completed():
                    is_terminal = True
                    reward = 1.
            else:
                reward -= 0.15

        # else:
        #     # Update puzzle pieces with action
        #     for puzzle in self.puzzles:
        #         if puzzle.update_puzzle(action):
        #             reward += 0.25  # Increase reward if puzzle succeeded with provided action
        #             puzzle_position = puzzle.get_position()
        #             self.map[puzzle_position[0], puzzle_position[1]] = puzzle.get_current_state()
        #
        #             # Update dependants
        #             depends_on = puzzle.get_depends_on()
        #             if depends_on:
        #                 puzzle_position = depends_on.get_position()
        #                 self.map[puzzle_position[0], puzzle_position[1]] = depends_on.get_current_state()
        #
        #             # Terminate if the terminal puzzle is reached and completed
        #             if puzzle is self.terminal_puzzle and puzzle.is_completed():
        #                 is_terminal = True
        #                 reward = 1.
        #                 break
        #         else:
        #             reward -= 0.15

        print(f"action: {action}, reward: {reward}, terminal: {is_terminal}")
        return reward, is_terminal

    def update_selected_puzzle(self):
        if self.map[self.current_grid_pos_x, self.current_grid_pos_y] != -1:
            for puzzle in self.puzzles:
                if puzzle.get_position() == [self.current_grid_pos_x, self.current_grid_pos_y]:
                    self.selected_puzzle = puzzle
                    break

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

    def get_position_map(self):
        position_map = np.full((self.map_width, self.map_height), 0.)
        position_map[self.current_grid_pos_x, self.current_grid_pos_y] = 1.
        # print("\n --------- POSITION MAP ---------")
        # print(position_map)
        return position_map


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

# ----------------------------- NEURAL NETWORK -----------------------------
class DQN(nn.Module):

    def __init__(self, num_actions=3):
        super(DQN, self).__init__()
        self.num_actions = num_actions
        self.gamma = 0.99
        self.final_epsilon = 0.0001
        self.initial_epsilon = 0.1
        self.num_iterations = 10000
        self.replay_memory_size = 500
        self.minibatch_size = 32

        self.layer1 = nn.Linear(4, 32, dtype=torch.float64)
        self.relu1 = nn.ReLU(inplace=True)
        self.layer2 = nn.Linear(256, 512, dtype=torch.float64)
        self.relu2 = nn.ReLU(inplace=True)
        self.layer3 = nn.Linear(512, self.num_actions, dtype=torch.float64)

    def forward(self, x):
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


def train(model, start, losses, q_values, completions):
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
    # print(game_state.get_map())
    # print("\n")

    action = torch.zeros([model.num_actions], dtype=torch.float64)
    action[0] = 1
    if torch.cuda.is_available():
        action = action.cuda()

    reward, terminal = game_state.step(action)
    state = torch.cat((torch.tensor(game_state.get_map()), torch.tensor(game_state.get_position_map()))).unsqueeze(0)
    # print("\n ------------- STATE ------------")
    # print(state)

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
        state_ = torch.cat((torch.tensor(game_state.get_map()),
                            torch.tensor(game_state.get_position_map()))).unsqueeze(0)
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
        # if iteration % 50 == 0:
        #     torch.save(model, "pretrained_model/current_model_" + str(iteration) + ".pth")

        print("iteration:", iteration, "elapsed time:", time.time() - start, "epsilon:", epsilon, "action:",
              action_index.cpu().detach().numpy(), "reward:", reward.numpy()[0][0], "Q max:",
              np.max(output.cpu().detach().numpy()))

        if terminal:
            completions.append(iteration)

            # initialise the map!
            button = Button([1, 1])
            door = Door([2, 2], button)

            game_state = GameState(4, 4)
            game_state.set_terminal_puzzle(door)

            game_state.add_puzzle(button)
            game_state.add_puzzle(door)
            state = torch.cat((torch.tensor(game_state.get_map()),
                               torch.tensor(game_state.get_position_map()))).unsqueeze(0)
            if torch.cuda.is_available():
                state = state.cuda()


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
        completions = []

        train(model, start, losses, q_values, completions)

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


if __name__ == "__main__":
    main(sys.argv[1])
