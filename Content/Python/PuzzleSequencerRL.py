import copy
import os.path
import shutil

import PuzzleSequencerGameState as PSGS
import PuzzleSequencerPuzzles as PSP

# import unreal # Disabled print :c
import random
import sys
from datetime import datetime
import logging
from collections import namedtuple

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib
import matplotlib.pyplot as plt


# ----------------------------- NEURAL NETWORK MODEL -----------------------------
class DQN(nn.Module):
    def __init__(self, learning_rate, input_dims, layer1_dims, layer2_dims, n_actions):
        super(DQN, self).__init__()
        self.learning_rate = learning_rate
        self.input_dims = input_dims
        self.layer1_dims = layer1_dims
        self.layer2_dims = layer2_dims
        self.n_actions = n_actions

        # Layer Set-up
        self.layer1 = nn.Linear(*self.input_dims, self.layer1_dims)
        self.relu1 = nn.ReLU(inplace=True)
        self.layer2 = nn.Linear(self.layer1_dims, self.layer2_dims)
        self.relu2 = nn.ReLU(inplace=True)
        self.layer3 = nn.Linear(self.layer2_dims, self.n_actions)

        self.optimiser = optim.Adam(self.parameters(), lr=learning_rate)
        # self.optimiser = optim.RMSprop(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu1(out)
        out = self.layer2(out)
        out = self.relu2(out)
        out = self.layer3(out)
        return out


# ------------------------------------- AGENT ------------------------------------
class Agent:
    layer1_dims = 256
    layer2_dims = 256

    def __init__(self, gamma, epsilon, learning_rate, input_dims, batch_size, n_actions,
                 max_memory_size=100000, target_model_update_rate=1000, epsilon_end=0.01, epsilon_dec=5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_end = epsilon_end
        self.epsilon_dec = epsilon_dec
        self.learning_rate = learning_rate
        self.input_dims = input_dims
        self.action_space = [i for i in range(n_actions)]
        self.batch_size = batch_size
        self.memory_size = max_memory_size
        self.target_model_update_rate = target_model_update_rate
        self.memory_cntr = 0
        self.iterations = 0

        self.model = DQN(learning_rate, input_dims, self.layer1_dims, self.layer2_dims, n_actions)
        self.target_model = DQN(learning_rate, input_dims, self.layer1_dims, self.layer2_dims, n_actions)

        # Replay Memory
        self.state_memory = np.zeros((self.memory_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.memory_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.memory_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.memory_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.memory_size, dtype=bool)

    def store_transition(self, action, state, state_, reward, terminal):
        index = self.memory_cntr % self.memory_size
        self.action_memory[index] = action
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.terminal_memory[index] = terminal

        self.memory_cntr += 1

    def choose_action(self, map_state):
        # Exploitation
        if np.random.random() > self.epsilon:
            state = torch.from_numpy(map_state).to(self.model.device)
            actions = self.model.forward(state)
            action = torch.argmax(actions).item()
        # Exploration
        else:
            action = np.random.choice(self.action_space)

        return action

    def set_model(self, model):
        self.model = model

    def learn(self):
        # Early-out batch is insufficient
        if self.memory_cntr < self.batch_size:
            return

        # self.model.optimiser.zero_grad()

        max_memory = min(self.memory_cntr, self.memory_size)

        batch = np.random.choice(max_memory, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        action_batch = self.action_memory[batch]
        state_batch = torch.tensor(self.state_memory[batch], device=self.model.device)
        new_state_batch = torch.tensor(self.new_state_memory[batch], device=self.model.device)
        reward_batch = torch.tensor(self.reward_memory[batch], device=self.model.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch], device=self.model.device)

        # Calculate Q
        q_value = self.model.forward(state_batch)[batch_index, action_batch]
        # q_next_value = self.model.forward(new_state_batch)  # this can be turned into target network to stabilise
        q_next_value = self.target_model(new_state_batch)

        q_next_value[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * torch.max(q_next_value, dim=1)[0]

        # Calculate Loss
        loss = self.model.loss(q_target, q_value).to(self.model.device)
        loss.backward()
        self.model.optimiser.step()

        self.epsilon = self.epsilon - self.epsilon_dec if self.epsilon > self.epsilon_end \
            else self.epsilon_end

        if self.iterations % self.target_model_update_rate == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        self.iterations += 1


# ---------------------------------- GAME STATE ----------------------------------
def initialise_game_state(randomise=False):
    game_state = PSGS.GameState(4, 4)

    if randomise:
        x, y = randomise_puzzle_location(game_state)
        ez_door = PSP.EzDoor([x, y])
        game_state.add_puzzle(ez_door)
        game_state.set_terminal_puzzle(ez_door)
        # x, y = randomise_puzzle_location(game_state)
        # button = PSP.Button([x, y])
        # game_state.add_puzzle(button)
        #
        # # x, y = randomise_puzzle_location(game_state)
        # # pressure_plate = PSP.PressurePlate([x, y], button)
        # # game_state.add_puzzle(pressure_plate)
        #
        # x, y = randomise_puzzle_location(game_state)
        # door = PSP.Door([x, y], button)
        # game_state.add_puzzle(door)
        #
        # game_state.set_terminal_puzzle(door)

    else:
        ez_door = PSP.EzDoor([1, 1])
        game_state.add_puzzle(ez_door)
        game_state.set_terminal_puzzle(ez_door)
        # # Init puzzles
        # button = PSP.Button([1, 1])
        # # pressure_plate = PSP.PressurePlate([3, 2], button)
        # door = PSP.Door([3, 3], button)
        #
        # # Set terminal puzzle
        # game_state.set_terminal_puzzle(door)
        #
        # # Add puzzles to game state
        # game_state.add_puzzle(button)
        # # game_state.add_puzzle(pressure_plate)
        # game_state.add_puzzle(door)
        # # print(game_state.get_map())

    # print("Available actions for the game state:")
    # print(PSGS.GameState.available_actions)
    # print("\n")
    return game_state


def randomise_puzzle_location(game_state):
    puzzle_map = game_state.get_map()

    rand_x = random.randint(0, game_state.get_map_width() - 1)
    rand_y = random.randint(0, game_state.get_map_height() - 1)
    while puzzle_map[rand_x, rand_y] != -1:
        rand_x = random.randint(0, game_state.get_map_width() - 1)
        rand_y = random.randint(0, game_state.get_map_height() - 1)

    return [rand_x, rand_y]


# -------------------------------------- TEST-------------------------------------
def test():
    game_state = initialise_game_state()
    agent = Agent(gamma=0.99, epsilon=0.01, batch_size=32, n_actions=len(game_state.available_actions),
                  input_dims=[32], learning_rate=0.0001, epsilon_end=0.01, epsilon_dec=2e-4,
                  target_model_update_rate=500)
    model = load_model("Pre-Trained Models/model_episode_140.pth", game_state)
    agent.set_model(model)

    n_games = 10000

    total_iterations = 0
    scores, average_scores = [], []

    # Disable gradient computation for model evaluation
    with torch.no_grad():
        for episode in range(n_games):
            score = 0
            iterations = 0
            terminal = False
            map_state = game_state.get_map_state()

            while not terminal:
                action = agent.choose_action(map_state)
                reward, terminal = game_state.step(action)

                new_map_state = game_state.get_map_state()

                score += reward
                iterations += 1

                map_state = new_map_state
                print(f"Iteration: {iterations} Action: {action} Reward: {reward} Score: {score} Terminal: {terminal}")

            scores.append(score)
            average_score = np.mean(scores[-10:])
            average_scores.append(average_score)

            print(f"Episode: [{episode}/{n_games}] \
                    Score: {score:.2f} \
                    Average Score: {average_score:.2f} \
                    Iterations: {iterations}")

            total_iterations += iterations
            game_state = initialise_game_state()
    print(f"TOTAL ITERATIONS: {total_iterations}")


# ------------------------------------- TRAIN ------------------------------------
def train():
    game_state = initialise_game_state()
    cached_game_state = copy.copy(game_state)

    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=32, n_actions=len(game_state.available_actions),
                  input_dims=[32], learning_rate=0.0001, epsilon_end=0.01, epsilon_dec=2e-5,
                  target_model_update_rate=500)

    n_games = 1000  # amount of complete game episodes
    model_save_rate = 10

    initialise_model_saving()
    initialise_logging(agent, n_games)

    total_iterations = 0
    scores, average_scores, epsilon_history = [], [], []

    for episode in range(n_games):
        score = 0
        terminal = False
        iterations = 0
        map_state = game_state.get_map_state()

        # Continue until game reached terminal state
        while not terminal:
            action = agent.choose_action(map_state)
            reward, terminal = game_state.step(action)
            new_map_state = game_state.get_map_state()

            score += reward
            iterations += 1

            agent.store_transition(action, map_state, new_map_state, reward, terminal)
            agent.learn()

            map_state = new_map_state

            print(
                f"Iteration: {iterations} \
                Action: {action} \
                Reward: {reward} \
                Score: {score} \
                Terminal: {terminal} \
                Epsilon: {agent.epsilon}")

            if iterations >= 500:
                break

        # Collect episode data
        scores.append(score)
        epsilon_history.append(agent.epsilon)

        average_score = np.mean(scores[-10:])
        average_scores.append(average_score)

        logging.info(f"Episode: {episode} \
        Score: {score:.2f} \
        Average Score: {average_score:.2f} \
        Epsilon: {agent.epsilon:.2f} \
        Iterations: {iterations}")

        total_iterations += iterations

        # Save model
        if episode % model_save_rate == 0:
            save_model(agent.model, episode)

        # Reset Game State
        game_state = initialise_game_state(True)

        # if agent.epsilon > 0.5 or episode % 10 == 0:
        #     # Reset Game State
        #     game_state = initialise_game_state(True)
        #     cached_game_state = copy.copy(game_state)
        # else:
        #     game_state = copy.copy(cached_game_state)

    logging.info(f"\nTOTAL ITERATIONS COMPLETED: {total_iterations}")

    current_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    save_log(current_time)

    # Plot data
    x = [i + 1 for i in range(n_games)]
    plot_learning(x, scores, epsilon_history, f"Output_Graph_EpsScore_{current_time}.png")


# ------------------------------------- PLOT -------------------------------------
def plot_learning(x, scores, epsilons, filename):
    fig = plt.figure(figsize=(15, 10), dpi=100)
    ax = fig.add_subplot(111, label="Epsilon")
    ax2 = fig.add_subplot(111, label="Score", frame_on=False)

    # Plot Epsilon
    ax.plot(x, epsilons, color='C0', label="Epsilon", linestyle='--', markerfacecolor="C0")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon", color='C0')
    ax.tick_params(axis='y', color='C0')
    ax.spines['left'].set_color('C0')

    # averaging window
    window = 20

    # Running average
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t - window):(t + 1)])

    # Exponential Moving Average
    ema = numpy_ewma_vectorized_v2(np.array(scores), window)

    # Plot running and exponential average + raw score
    ax2.plot(x, running_avg, color='orange', linestyle='solid', marker='o', markerfacecolor='orange', markersize=8,
             label="Running Average Score")
    ax2.plot(x, ema, color='green', linestyle='solid', marker='o', markerfacecolor='green', markersize=8,
             label="Exponential Moving Average Score")
    ax2.plot(x, scores, color='red', linestyle='solid', marker='o', markerfacecolor='red', markersize=8,
             label="Raw Score")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position('right')
    ax2.set_ylabel("Score")

    # Attach legend on top of graph
    fig.legend(loc="upper center", ncol=2)
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)

    # Save graph
    save_path = os.path.dirname(os.path.abspath(__file__)) + "/Graphs/"
    if not os.path.exists(save_path):
        os.makedirs("Graphs")

    plt.savefig(save_path + filename, bbox_inches="tight", dpi=100)

    # Show Graph
    plt.show(block=True)


def plot_testing():
    fig = plt.figure(figsize=(15, 10), dpi=100)
    ax = fig.add_subplot(111, label="")


# https://localcoder.org/numpy-version-of-exponential-weighted-moving-average-equivalent-to-pandas-ewm
def numpy_ewma_vectorized_v2(data, window):
    alpha = 2 / (window + 1.0)
    alpha_rev = 1 - alpha
    n = data.shape[0]

    pows = alpha_rev ** (np.arange(n + 1))

    scale_arr = 1 / pows[:-1]
    offset = data[0] * pows[1:]
    pw0 = alpha * alpha_rev ** (n - 1)

    mult = data * pw0 * scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums * scale_arr[::-1]
    return out


# ------------------------------------- LOGGING ----------------------------------
def initialise_logging(agent, episodes):
    save_path = os.path.dirname(os.path.abspath(__file__)) + "/Logs/"
    if not os.path.exists(save_path):
        os.makedirs("Logs")
    logging.basicConfig(filename=save_path + f'output_recent.txt', level=logging.INFO, format='', filemode='w')

    logging.info(f"Episodes: {episodes} | Gamma: {agent.gamma} | Learning Rate: {agent.learning_rate} \
    Epsilon: {agent.epsilon} | Epsilon Decrement: {agent.epsilon_dec} | Epsilon End: {agent.epsilon_end} \
    Batch Size: {agent.batch_size} | Number Of Actions: {len(agent.action_space)} | Input Dimensions: {agent.input_dims} \
    Target Network Update Rate: {agent.target_model_update_rate}\n")


def save_log(current_time):
    save_path = os.path.dirname(os.path.abspath(__file__)) + "/Logs/"
    shutil.copyfile(save_path + f'output_recent.txt', save_path + f'output_{current_time}.txt')


def initialise_model_saving():
    save_path = os.path.dirname(os.path.abspath(__file__)) + "/Pre-Trained Models/"
    if not os.path.exists(save_path):
        os.makedirs("Pre-Trained Models")


# Save model's state dictionary for inference
def save_model(model, episode):
    torch.save(model.state_dict(), f"Pre-Trained Models/model_episode_{str(episode)}.pth")


# Load model's state dictionary for inference
def load_model(model_path, game_state):
    model = DQN(0.001, [32], 256, 256, len(game_state.available_actions))
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


# ------------------------------------- MAIN -------------------------------------
def main(mode):
    if mode == "train":
        train()

    elif mode == "test":
        test()


if __name__ == "__main__":
    main(sys.argv[1])
