import numpy as np
from keras import Sequential


class Memory(object):

    def __init__(self, env_dim: int, max_memory: int, discount: float):
        self.__max_memory = max_memory
        self.__memory = list()
        self.__discount = discount
        self.__env_dim = env_dim

    def remember(self, states: list, is_done: bool):
        self.__memory.append([states, is_done])
        if len(self.__memory) > self.__max_memory:
            del self.__memory[0]

    def get_batch(self, model: Sequential, max_batch_size=10):
        num_actions = model.output_shape[-1]

        len_memory = len(self.__memory)
        batch_size = min(len_memory, max_batch_size)

        inputs = np.zeros((batch_size, self.__env_dim))
        targets = np.zeros((batch_size, num_actions))

        for i, idx in enumerate(list(np.random.randint(0, len_memory, size=batch_size))):
            state_t, action_t, reward_t, state_tp1 = self.__memory[idx][0]

            game_over = self.__memory[idx][1]
            inputs[i:i + 1] = state_t

            targets[i] = model.predict(state_t)[0]

            q_sa = np.max(model.predict(state_tp1)[0])

            if game_over:
                targets[i, action_t] = reward_t
            else:
                targets[i, action_t] = reward_t + self.__discount * q_sa

        return inputs, targets
