import PuzzleSequencerGameState as PSGS


class PuzzleObject:
    def __init__(self, position, available_states):
        self.current_state = 0
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
        super().__init__(position, {"unpressed": 0, "pressed": 1})
        PSGS.GameState.add_available_action_as_key("press")

    def update_puzzle(self, action):
        reward = 0.

        if self.completed:
            return reward

        # checking the one hot encoding
        if ((action == 1.).nonzero(as_tuple=True)[0]) == PSGS.GameState.available_actions["press"]:
            self.current_state = self.available_states["pressed"]
            self.completed = True
            reward = 0.25

        # punish
        else:
            reward = -0.15

        return reward


class PressurePlate(PuzzleObject):
    def __init__(self, position, depends_on):
        super().__init__(position, {"de-activated": 0, "activated": 1})
        PSGS.GameState.add_available_action_as_key("activate")
        self.depends_on = depends_on

    def update_puzzle(self, action):
        reward = 0.

        if self.completed:
            return reward

        if self.depends_on.is_completed() and \
                ((action == 1.).nonzero(as_tuple=True)[0]) == PSGS.GameState.available_actions["activate"]:
            self.current_state = self.available_states["activated"]
            self.completed = True
            reward = 0.25

        # punish
        else:
            reward = -0.15

        return reward


class Door(PuzzleObject):
    def __init__(self, position, depends_on):
        super().__init__(position, {"locked": 0, "closed": 1, "open": 2})
        PSGS.GameState.add_available_action_as_key("open")
        self.depends_on = depends_on

    def update_puzzle(self, action):
        reward = 0.

        # Early-out completed puzzle
        if self.completed:
            return reward

        # Unlock door if dependant object is completed
        if self.depends_on.is_completed() and self.current_state is self.available_states["locked"]:
            self.current_state = self.available_states["closed"]

        # Open unlocked door
        if self.current_state == self.available_states["closed"] and \
                ((action == 1.).nonzero(as_tuple=True)[0]) == PSGS.GameState.available_actions["open"]:
            self.current_state = self.available_states["open"]
            self.completed = True
            reward = 0.25

        # punish
        else:
            reward = -0.15

        return reward
