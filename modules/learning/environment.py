class DdosClassifierEnv:

    def __init__(self):
        self.state = [0, 0, 0, 0, 0]

    def step(self, action):
        match action:
            case (0):
                self.state[0] += 1
            case (1):
                self.state[0] -= 1
            case (2):
                self.state[1] += 1
            case (3):
                self.state[1] -= 1
            case (4):
                self.state[2] += 1
            case (5):
                self.state[2] -= 1
            case (6):
                self.state[3] += 1
            case (7):
                self.state[3] -= 1
            case (8):
                self.state[4] += 1
            case (9):
                self.state[4] -= 1
            case _:
                pass
