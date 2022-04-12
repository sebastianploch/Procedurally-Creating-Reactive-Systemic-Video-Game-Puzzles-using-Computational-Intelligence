import os
import sys
import random
import logging
import shutil
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

import PuzzleSequencerGameState as PSGS
import PuzzleSequencerPuzzles as PSP


class ReplayBuffer:
    def __init__(self, max_size, input_shape):
        self.memory_size = max_size
        self.memory_counter = 0

        self.state_memory = np.zeros((self.memory_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.memory_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.memory_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.memory_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.memory_size, dtype=bool)

    def store_transition(self, state, state_, action, reward, terminal):
        index = self.memory_counter % self.memory_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = terminal

        self.memory_counter += 1

    def sample_buffer(self, batch_size):
        max_memory = min(self.memory_counter, self.memory_size)
        batch = np.random.choice(max_memory, batch_size, replace=False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminals = self.terminal_memory[batch]

        return states, states_, actions, rewards, terminals


class DDQN(nn.Module):
    def __init__(self, learning_rate, n_actions, name, input_dims, checkpoint_dir):
        super(DDQN, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.layer1 = nn.Linear(*input_dims, 512)
        self.value = nn.Linear(512, 1)  # Value stream
        self.advantage = nn.Linear(512, n_actions)  # Advantage action stream

        self.optimiser = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()

        # Decide whether to run on GPU or CPU
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        flat_layer1 = F.relu(self.layer1(state))
        value = self.value(flat_layer1)
        advantage = self.advantage(flat_layer1)

        return value, advantage

    def save_checkpoint(self):
        print("... Saving Checkpoint ...")
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("... Loading Checkpoint ...")
        self.load_state_dict(torch.load(self.checkpoint_file))


class Agent:
    def __init__(self, gamma, learning_rate, epsilon, epsilon_min, epsilon_decrement,
                 n_actions, input_dims, memory_size, batch_size, target_network_replace=1000,
                 checkpoint_dir="Pre-Trained Models DDQN"):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decrement = epsilon_decrement
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.target_network_replace_counter = target_network_replace
        self.checkpoint_dir = checkpoint_dir
        self.learn_step_counter = 0

        self.action_space = [i for i in range(self.n_actions)]
        self.memory = ReplayBuffer(memory_size, input_dims)

        self.q_eval = DDQN(self.learning_rate, self.n_actions, input_dims=self.input_dims,
                           name="DDQN_Eval", checkpoint_dir=self.checkpoint_dir)
        self.q_next = DDQN(self.learning_rate, self.n_actions, input_dims=self.input_dims,
                           name="DDQN_Next", checkpoint_dir=self.checkpoint_dir)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:  # Greedy Action
            state = torch.tensor([observation], dtype=torch.float32, device=self.q_eval.device)
            _, advantage = self.q_eval.forward(state)
            action = torch.argmax(advantage).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def store_transition(self, state, state_, action, reward, terminal):
        self.memory.store_transition(state, state_, action, reward, terminal)

    def replace_target_network(self):
        if self.learn_step_counter % self.target_network_replace_counter == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.epsilon_decrement \
            if self.epsilon > self.epsilon_min else self.epsilon_min

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        if self.memory.memory_counter < self.batch_size:
            return

        self.q_eval.optimiser.zero_grad()
        self.replace_target_network()

        state, new_state, action, reward, terminal = self.memory.sample_buffer(self.batch_size)

        states = torch.tensor(state, device=self.q_eval.device)
        new_states = torch.tensor(new_state, device=self.q_eval.device)
        actions = torch.tensor(action, device=self.q_eval.device)
        rewards = torch.tensor(reward, device=self.q_eval.device)
        terminals = torch.tensor(terminal, device=self.q_eval.device)

        indices = np.arange(self.batch_size)

        value_state, advantage_state = self.q_eval.forward(states)
        value_state_, advantage_state_ = self.q_next.forward(new_states)
        value_state_eval, advantage_state_eval = self.q_eval.forward(new_states)

        q_pred = torch.add(value_state,
                           (advantage_state - advantage_state.mean(dim=1, keepdim=True)))[indices, actions]
        q_next = torch.add(value_state_,
                           (advantage_state_ - advantage_state_.mean(dim=1, keepdim=True)))
        q_eval = torch.add(value_state_eval,
                           (advantage_state_eval - advantage_state_eval.mean(dim=1, keepdim=True)))

        max_actions = torch.argmax(q_eval, dim=1)
        q_next[terminals] = 0.0
        q_target = rewards + self.gamma * q_next[indices, max_actions]

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimiser.step()
        self.learn_step_counter += 1
        self.decrement_epsilon()


def train():
    game_state = initialise_game_state()

    agent = Agent(gamma=0.99, learning_rate=5e-4, epsilon=1.0, epsilon_min=0.01, epsilon_decrement=1e-3,
                  n_actions=len(PSGS.GameState.available_actions), input_dims=[32], memory_size=1000000,
                  batch_size=64, target_network_replace=1000)

    num_episodes = 500
    scores, epsilon_history = [], []
    total_iterations = 0

    initialise_logging(agent, num_episodes)

    for i in range(num_episodes):
        terminal = False
        observation = game_state.get_map_state()
        score = 0
        iterations = 0

        while not terminal:
            action = agent.choose_action(observation)
            reward, terminal = game_state.step(action)
            observation_ = game_state.get_map_state()

            score += reward
            iterations += 1

            agent.store_transition(observation, observation_, action, reward, terminal)
            agent.learn()

            observation = observation_

            print(f"Iteration: {iterations} \
                  Action: {action} \
                  Reward: {reward} \
                  Score: {score} \
                  Terminal: {terminal} \
                  Epsilon: {agent.epsilon}")

        epsilon_history.append(agent.epsilon)
        scores.append(score)
        average_score = np.mean(scores[-100:])

        logging.info(f"Episode: {i} \
        Score: {score:.2f} \
        Average Score: {average_score:.2f} \
        Epsilon: {agent.epsilon:.2f} \
        Iterations: {iterations}")

        total_iterations += iterations

        game_state = initialise_game_state(True)

    logging.info(f"\nTOTAL ITERATIONS COMPLETED: {total_iterations}")

    current_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    save_log(current_time)

    # Plot data
    x = [i + 1 for i in range(num_episodes)]
    plot_learning(x, scores, epsilon_history, f"Output_Graph_EpsScore_{current_time}.png")


def test():
    pass


def initialise_game_state(randomise=False):
    game_state = PSGS.GameState(4, 4)

    if randomise:
        x, y = randomise_puzzle_location(game_state)
        ez_door = PSP.EzDoor([x, y])
        game_state.add_puzzle(ez_door)
        game_state.set_terminal_puzzle(ez_door)
    else:
        ez_door = PSP.EzDoor([1, 1])
        game_state.add_puzzle(ez_door)
        game_state.set_terminal_puzzle(ez_door)

    return game_state


def randomise_puzzle_location(game_state):
    puzzle_map = game_state.get_map()

    rand_x = random.randint(0, game_state.get_map_width() - 1)
    rand_y = random.randint(0, game_state.get_map_height() - 1)
    while puzzle_map[rand_x, rand_y] != -1:
        rand_x = random.randint(0, game_state.get_map_width() - 1)
        rand_y = random.randint(0, game_state.get_map_height() - 1)

    return [rand_x, rand_y]


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
    save_path = os.path.dirname(os.path.abspath(__file__)) + "/Graphs DDQN/"
    if not os.path.exists(save_path):
        os.makedirs("Graphs DDQN")

    plt.savefig(save_path + filename, bbox_inches="tight", dpi=100)

    # Show Graph
    plt.show(block=True)


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


def initialise_logging(agent, episodes):
    save_path = os.path.dirname(os.path.abspath(__file__)) + "/Logs DDQN/"
    if not os.path.exists(save_path):
        os.makedirs("Logs DDQN")
    logging.basicConfig(filename=save_path + f'output_recent.txt', level=logging.INFO, format='', filemode='w')

    logging.info(f"Episodes: {episodes} | Gamma: {agent.gamma} | Learning Rate: {agent.learning_rate} \
    Epsilon: {agent.epsilon} | Epsilon Decrement: {agent.epsilon_decrement} | Epsilon End: {agent.epsilon_min} \
    Batch Size: {agent.batch_size} | Number Of Actions: {len(agent.action_space)} | Input Dimensions: {agent.input_dims} \
    Target Network Update Rate: {agent.target_network_replace_counter}\n")


def save_log(current_time):
    save_path = os.path.dirname(os.path.abspath(__file__)) + "/Logs DDQN/"
    shutil.copyfile(save_path + f'output_recent.txt', save_path + f'output_{current_time}.txt')


def main(mode):
    if mode == "train":
        train()

    elif mode == "test":
        test()


if __name__ == "__main__":
    main(sys.argv[1])
