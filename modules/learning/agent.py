import numpy as np
from keras import Sequential
from tqdm import trange

from modules.learning.environment import ClassifierEnv
from modules.learning.memory import Memory


class Agent:

    def __init__(self, env: ClassifierEnv, memory: Memory, model: Sequential, epochs: int, epsilon: float, batch_size: int,
                 time_steps: int):
        self.loss_list = []
        self.reward_list = []
        self.filters_history = []

        self.__env = env
        self.__memory = memory
        self.__model = model
        self.__epochs = epochs
        self.__epsilon = epsilon
        self.__batch_size = batch_size
        self.__time_steps = time_steps

    def train(self) -> None:
        for _ in trange(self.__epochs, desc='Номер эпохи'):
            observation = self.__env.reset()

            total_loss = 0
            total_reward = 0

            for _ in trange(self.__time_steps, desc='Номер данных'):
                state_before = np.array(observation, ndmin=2)

                if np.random.rand() < self.__epsilon:
                    action = self.__env.action_space.sample()
                else:
                    action = np.argmax(self.__model.predict(state_before, verbose=0)[0])

                observation, reward, is_done = self.__env.step(action)
                total_reward += reward

                state_after = np.array(observation, ndmin=2)

                self.__memory.remember([state_before, action, reward, state_after], is_done)

                inputs, targets = self.__memory.get_batch(self.__model, self.__batch_size)

                loss = self.__model.train_on_batch(inputs, targets)
                total_loss += loss

                if is_done:
                    break

            self.loss_list.append(total_loss)
            self.reward_list.append(total_reward)

            self.filters_history.append(repr(self.__env.filters))

        self.__env.reset()

    def test(self, games: int = 2, time_steps: int = 10) -> None:
        for _ in range(games):
            observation = self.__env.reset()

            total_reward = 0
            steps = 0

            for _ in range(time_steps):
                steps += 1

                state = np.array(observation, ndmin=2)
                action = np.argmax(self.__model.predict(state)[0])

                observation, reward, is_done = self.__env.step(action)
                total_reward += reward

                if is_done:
                    break

            print(total_reward / steps)
