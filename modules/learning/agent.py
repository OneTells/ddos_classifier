import numpy as np
from keras import Sequential
from tqdm import trange

from modules.learning.environment import ClassifierEnv
from modules.learning.memory import Memory

bar_format = '{desc}:{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed_s:4.0f}s прошло | {remaining_s:4.0f}s осталось]'


class Agent:

    def __init__(self, model: Sequential) -> None:
        self.loss_list = []
        self.reward_list = []
        self.filters_history = []

        self.__model = model

    def train(self, env: ClassifierEnv, memory: Memory, epochs: int, epsilon: float, batch_size: int, time_steps: int) -> None:
        for number_epoch in trange(epochs, desc='Обучение модели', bar_format=bar_format):
            observation = env.reset()

            total_loss = 0
            total_reward = 0

            for _ in trange(time_steps, desc=f'Обучение {number_epoch + 1} эпохи', bar_format=bar_format):
                state_before = np.array(observation, ndmin=2)

                if np.random.rand() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(self.__model.predict(state_before, verbose=0)[0])

                observation, reward, is_done = env.step(action)
                total_reward += reward

                state_after = np.array(observation, ndmin=2)

                memory.remember([state_before, action, reward, state_after], is_done)

                inputs, targets = memory.get_batch(self.__model, batch_size)

                loss = self.__model.train_on_batch(inputs, targets)
                total_loss += loss

                if is_done:
                    break

            self.loss_list.append(total_loss)
            self.reward_list.append(total_reward)

            self.filters_history.append(repr(env.filters))
            self.__model.save_weights(f'save_point_{number_epoch + 1}.weights.h5')

        env.reset()

    def test(self, env: ClassifierEnv, epochs: int = 2, time_steps: int = 10) -> None:
        for _ in range(epochs):
            observation = env.reset()

            total_reward = 0
            steps = 0

            for _ in range(time_steps):
                steps += 1

                state = np.array(observation, ndmin=2)
                action = np.argmax(self.__model.predict(state)[0])

                observation, reward, is_done = env.step(action)
                total_reward += reward

                if is_done:
                    break

            print(total_reward / steps)
