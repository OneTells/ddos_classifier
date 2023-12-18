import numpy as np

from modules.learning.memory import Memory


class Agent:

    def __init__(self, env, memory: Memory, model, epochs, epsilon=0.9, batch_size=10, time_steps=200):
        self.loss_list = []
        self.reward_list = []

        self.__env = env
        self.__memory = memory
        self.__model = model
        self.__epochs = epochs
        self.__epsilon = epsilon
        self.__batch_size = batch_size
        self.__time_steps = time_steps

    def train(self) -> None:
        for i_episode in range(self.__epochs):
            observation = self.__env.reset()

            loss_on_epochs = 0
            reward_on_epochs = 0

            for t in range(self.__time_steps):
                state_before = np.array(observation, ndmin=2)

                if np.random.rand() < self.__epsilon:
                    action = self.__env.action_space.sample()
                else:
                    action = np.argmax(self.__model.predict(state_before)[0])

                observation, reward, done = self.__env.step(action)
                reward_on_epochs += reward

                state_after = np.array(observation, ndmin=2)

                self.__memory.remember([state_before, action, reward, state_after], done)

                inputs, targets = self.__memory.get_batch(self.__model, max_batch_size=self.__batch_size)

                loss = self.__model.train_on_batch(inputs, targets)
                loss_on_epochs += loss

            self.loss_list.append(loss_on_epochs)
            self.reward_list.append(reward_on_epochs)

        self.__env.reset()

    def test(self, games: int = 2, time_steps: int = 10):
        for _ in range(games):
            observation = self.__env.reset()
            total_reward = 0

            for _ in range(time_steps):
                state = np.array(observation, ndmin=2)
                action = np.argmax(self.__model.predict(state)[0])
                observation, reward, done = self.__env.step(action)
                total_reward += reward

            print(total_reward / time_steps)
