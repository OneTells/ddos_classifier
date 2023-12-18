import numpy as np
from keras.models import Sequential
from keras.src.layers import Dense
from keras.src.optimizers import Adam

from modules.learning.environment import ClassifierEnv


class Memory(object):

    def __init__(self, env_dim, max_memory=100, discount=0.9):
        self.__max_memory = max_memory
        self.__memory = list()
        self.__discount = discount
        self.__env_dim = env_dim

    def remember(self, states, game_over):
        self.__memory.append([states, game_over])
        if len(self.__memory) > self.__max_memory:
            del self.__memory[0]

    def get_batch(self, model, max_batch_size=10):
        num_actions = model.output_shape[-1]

        len_memory = len(self.__memory)
        batch_size = min(len_memory, max_batch_size)

        inputs = np.zeros((batch_size, self.__env_dim))
        targets = np.zeros((batch_size, num_actions))

        for i, idx in enumerate(np.random.randint(0, len_memory, size=batch_size)):
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
        env = ClassifierEnv()

        for _ in range(games):
            observation = env.reset()
            total_reward = 0

            for _ in range(time_steps):
                state = np.array(observation, ndmin=2)
                action = np.argmax(self.__model.predict(state)[0])
                observation, reward, done = env.step(action)
                total_reward += reward

            print(total_reward / time_steps)

def create_model(hidden_size: float, feature_dims: int, num_actions: int, learning_rate: float):
    model = Sequential()
    model.add(Dense(hidden_size, input_shape=(feature_dims,), activation='relu'))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(num_actions))
    model.compile(Adam(learning_rate), "mse")
    return model


def main():
    epsilon = 0.10  # probability of choosing a random action instead of using the model to decide
    max_memory = 200  # max number of experiences to be stored at once
    hidden_size = 100  # size of the hidden layers within the network
    batch_size = 20  # amount of experiences to sample into each batch for training
    discount = 0.95  # value of future reward vs. current reward
    learning_rate = 0.005  # the multiplicative rate at which the weights of the model are shifted
    time_steps = 10  # length of each game (for Cartpole, ideally set this to between 100-200)
    epochs = 1  # (Amount of games played)

    env = ClassifierEnv()

    feature_dims = len(env.reset())
    num_actions = env.action_space.n

    memory = Memory(max_memory=max_memory, discount=discount, env_dim=feature_dims)

    model = create_model(hidden_size, feature_dims, num_actions, learning_rate)

    agent = Agent(env, memory, model, epochs, epsilon, batch_size, time_steps)
    agent.train()

    model.save_weights(f'model.weights.h5')

    # model.load_weights(f'model.weights.h5')
    # agent.test()


if __name__ == '__main__':
    main()
