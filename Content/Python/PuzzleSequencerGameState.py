import math
import numpy as np

import PuzzleSequencerConstants as PSC


class GameState:
    available_actions = {"idle": 0, "move_up": 1, "move_down": 2, "move_left": 3, "move_right": 4}
    """All available actions in the world"""

    def __init__(self, map_width, map_height):
        self.puzzles = []
        self.starting_puzzle = None
        self.terminal_puzzle = None
        self.map_width = map_width
        self.map_height = map_height
        self.map = np.full((map_width, map_height), -1)

        self.invalid_actions_taken = 0
        self.current_grid_pos_x = 0
        self.current_grid_pos_y = 0
        self.selected_puzzle = None
        self.goal_puzzle = None
        # self.iterations = 0

    @staticmethod
    def add_available_action_as_dict(action: dict[str, int]):
        for key, val in action.items():
            if key not in GameState.available_actions.keys():
                if val not in GameState.available_actions.values():
                    GameState.available_actions.update({key: val})
                else:
                    print(f"Value: {val} for Key: {key} is already present in available actions, pick different value!")

    @staticmethod
    def add_available_action_as_key(action: str):
        # Early-out, action already present
        if action in GameState.available_actions.keys():
            return

        # Find value for new action
        value = GameState.available_actions[max(GameState.available_actions, key=GameState.available_actions.get)]
        value += 1

        GameState.available_actions.update({action: value})

    def step(self, action):
        is_terminal = False
        reward = 0
        # self.iterations += 1

        # Early-out idle action
        if action == GameState.available_actions["idle"]:
            reward = PSC.IDLE_MOVE_REWARD
            self.invalid_actions_taken += 1

        # Select grid
        elif action == GameState.available_actions["move_up"]:
            if self.current_grid_pos_y - 1 >= 0:
                reward = PSC.VALID_MOVE_REWARD
                self.current_grid_pos_y -= 1
                self.update_selected_puzzle()
                # return reward, is_terminal
                self.invalid_actions_taken += 1
            else:
                reward = PSC.INVALID_MOVE_REWARD
                self.invalid_actions_taken += 1

        elif action == GameState.available_actions["move_down"]:
            if self.current_grid_pos_y + 1 < self.map_height:
                reward = PSC.VALID_MOVE_REWARD
                self.current_grid_pos_y += 1
                self.update_selected_puzzle()
                # return reward, is_terminal
                self.invalid_actions_taken += 1
            else:
                reward = PSC.INVALID_MOVE_REWARD
                self.invalid_actions_taken += 1

        elif action == GameState.available_actions["move_left"]:
            if self.current_grid_pos_x - 1 >= 0:
                reward = PSC.VALID_MOVE_REWARD
                self.current_grid_pos_x -= 1
                self.update_selected_puzzle()
                # return reward, is_terminal
                self.invalid_actions_taken += 1
            else:
                reward = PSC.VALID_MOVE_REWARD
                self.invalid_actions_taken += 1

        elif action == GameState.available_actions["move_right"]:
            if self.current_grid_pos_x + 1 < self.map_width:
                reward = PSC.VALID_MOVE_REWARD
                self.current_grid_pos_x += 1
                self.update_selected_puzzle()
                # return reward, is_terminal
                self.invalid_actions_taken += 1
            else:
                reward = PSC.INVALID_MOVE_REWARD
                self.invalid_actions_taken += 1

        # Update puzzle it is in the selected grid space
        elif self.selected_puzzle is not None:
            reward, valid_move = self.selected_puzzle.update(action)
            puzzle_position = self.selected_puzzle.get_position()
            self.map[puzzle_position[0], puzzle_position[1]] = self.selected_puzzle.get_current_state()

            # Update dependants
            depends_on = self.selected_puzzle.get_depends_on()
            if depends_on:
                puzzle_position = depends_on.get_position()
                self.map[puzzle_position[0], puzzle_position[1]] = depends_on.get_current_state()

            # Update Goal
            if self.selected_puzzle.is_completed():
                if self.selected_puzzle is self.goal_puzzle:
                    self.update_goal_puzzle()

                # Terminate if the terminal puzzle is reached and completed
                if self.selected_puzzle is self.terminal_puzzle:
                    is_terminal = True
                    reward = PSC.TERMINAL_PUZZLE_REWARD
                    return reward, is_terminal

            if valid_move:
                return reward, is_terminal
            else:
                self.invalid_actions_taken += 1

        else:
            reward = PSC.INVALID_PUZZLE_REWARD
            self.invalid_actions_taken += 1

        # Determine whether getting closer to goal or further and adjust reward accordingly
        goal_position = np.array(self.goal_puzzle.get_position())
        current_position = np.array([self.current_grid_pos_x, self.current_grid_pos_y])
        distance = np.linalg.norm(current_position - goal_position)
        reward -= distance

        # Apply reward discount depending on the amount of actions taken
        # reward -= PSC.INVALID_ACTION_TAKEN_DISCOUNT * self.invalid_actions_taken
        # reward = max(reward, 0.)
        # reward -= PSC.INVALID_ACTION_TAKEN_DISCOUNT * self.invalid_actions_taken

        # Normalise the reward using sigmoid
        # reward = 1 / (1 + np.exp(-reward))
        # reward = (reward - -1) / (1 - -1)

        return reward, is_terminal

    def reset(self):
        self.current_grid_pos_x = 0
        self.current_grid_pos_y = 0
        self.selected_puzzle = None
        self.goal_puzzle = self.starting_puzzle
        for puzzle in self.puzzles:
            position = puzzle.get_position()
            self.map[position[0], position[1]] = 0
            puzzle.reset()

    def update_selected_puzzle(self):
        if self.map[self.current_grid_pos_x, self.current_grid_pos_y] != -1:
            for puzzle in self.puzzles:
                if puzzle.get_position() == [self.current_grid_pos_x, self.current_grid_pos_y]:
                    self.selected_puzzle = puzzle
                    break

    def update_goal_puzzle(self):
        self.goal_puzzle = self.goal_puzzle.connects_to

    def add_puzzle(self, puzzle):
        puzzle_x = puzzle.get_position()[0]
        puzzle_y = puzzle.get_position()[1]
        self.map[puzzle_x, puzzle_y] = 0
        self.puzzles.append(puzzle)

    def set_starting_puzzle(self, puzzle):
        self.starting_puzzle = puzzle
        self.goal_puzzle = self.starting_puzzle

    def get_starting_puzzle(self):
        return self.starting_puzzle

    def set_terminal_puzzle(self, puzzle):
        self.terminal_puzzle = puzzle

    def get_terminal_puzzle(self):
        return self.terminal_puzzle

    def get_puzzles(self):
        return self.puzzles

    def get_map(self):
        return self.map

    def get_total_map_size(self):
        return self.map_width * self.map_height

    def get_map_size(self) -> tuple[int, int]:
        return self.map_width, self.map_height

    def get_map_width(self):
        return self.map_width

    def get_map_height(self):
        return self.map_height

    def get_position_map(self):
        position_map = np.full((self.map_width, self.map_height), 0)
        position_map[self.current_grid_pos_y, self.current_grid_pos_x] = 1
        return position_map

    def get_goal_map(self):
        goal_map = np.full((self.map_width, self.map_height), 0)
        goal_position = (self.goal_puzzle.get_position() if self.goal_puzzle else self.terminal_puzzle.get_position())
        goal_map[goal_position[0], goal_position[1]] = 1
        return goal_map

    def get_map_state(self):
        # ar1 = self.get_map()
        # ar2 = self.get_position_map()
        # ar3 = self.get_goal_map()
        # test = np.concatenate((self.get_map(), self.get_position_map(), self.get_goal_map()))
        array = np.array([self.get_map(), self.get_position_map(), self.get_goal_map()], dtype=np.float32).ravel()
        # normalise_iterations = -(self.iterations / 1000)
        # array = np.append(array, normalise_iterations)

        return array
