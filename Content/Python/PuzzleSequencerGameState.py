import numpy as np


# available_actions = {"idle": 0., "press": 1., "activate": 2., "open": 3., "place_pressure_plate": 4.,
#                      # Puzzle Interaction
#                      "move_up": 5., "move_down": 6., "move_left": 7., "move_right": 8.}  # Grid Selection
# """All available actions in the world"""


class GameState:
    available_actions = {"idle": 0, "move_up": 1, "move_down": 2, "move_left": 3, "move_right": 4}
    """All available actions in the world"""

    def __init__(self, map_width, map_height):
        self.puzzles = []
        self.terminal_puzzle = None
        self.map_width = map_width
        self.map_height = map_height
        self.map = np.full((map_width, map_height), -1)

        self.current_grid_pos_x = 0
        self.current_grid_pos_y = 0
        self.selected_puzzle = None

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
        reward = 0.

        # Early-out idle action
        if ((action == 1).nonzero(as_tuple=True)[0]) == GameState.available_actions["idle"]:
            return reward, is_terminal

        # Select grid
        if ((action == 1).nonzero(as_tuple=True)[0]) == GameState.available_actions["move_up"]:
            if self.current_grid_pos_y - 1 >= 0:
                reward += 0.01
                self.current_grid_pos_y -= 1
                self.update_selected_puzzle()
            else:
                reward -= 0.5
            return reward, is_terminal

        elif ((action == 1).nonzero(as_tuple=True)[0]) == GameState.available_actions["move_down"]:
            if self.current_grid_pos_y + 1 < self.map_height:
                reward += 0.01
                self.current_grid_pos_y += 1
                self.update_selected_puzzle()
            else:
                reward -= 0.5
            return reward, is_terminal

        elif ((action == 1).nonzero(as_tuple=True)[0]) == GameState.available_actions["move_left"]:
            if self.current_grid_pos_x - 1 >= 0:
                reward += 0.01
                self.current_grid_pos_x -= 1
                self.update_selected_puzzle()
            else:
                reward -= 0.5
            return reward, is_terminal

        elif ((action == 1).nonzero(as_tuple=True)[0]) == GameState.available_actions["move_right"]:
            if self.current_grid_pos_x + 1 < self.map_width:
                reward += 0.01
                self.current_grid_pos_x += 1
                self.update_selected_puzzle()
            else:
                reward -= 0.5
            return reward, is_terminal

        # Update puzzle it is in the selected grid space
        if self.selected_puzzle is not None:
            reward = self.selected_puzzle.update_puzzle(action)
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
        self.map[puzzle_x, puzzle_y] = 0
        self.puzzles.append(puzzle)

    def set_terminal_puzzle(self, puzzle):
        self.terminal_puzzle = puzzle

    def get_terminal_puzzle(self):
        return self.terminal_puzzle

    def get_map(self):
        return self.map

    def get_map_size(self):
        return self.map_width * self.map_height

    def get_puzzles(self):
        return self.puzzles

    def get_position_map(self):
        position_map = np.full((self.map_width, self.map_height), 0)
        position_map[self.current_grid_pos_y, self.current_grid_pos_x] = 1
        # print("\n --------- POSITION MAP ---------")
        # print(position_map)
        return position_map
