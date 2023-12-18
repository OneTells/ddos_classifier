import numpy as np
from keras.models import Sequential
from keras.src.layers import Dense
from keras.src.optimizers import Adam

from modules.learning.environment import ClassifierEnv


class Memory(object):

    def __init__(self, env_dim, max_memory=100, discount=.9):
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount
        self.env_dim = env_dim

    def remember(self, states, game_over):
        self.memory.append([states, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, max_batch_size=10):
        num_actions = model.output_shape[-1]

        len_memory = len(self.memory)
        batch_size = min(len_memory, max_batch_size)

        inputs = np.zeros((batch_size, self.env_dim))
        targets = np.zeros((batch_size, num_actions))

        for i, idx in enumerate(np.random.randint(0, len_memory, size=batch_size)):
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]

            game_over = self.memory[idx][1]
            inputs[i:i + 1] = state_t

            targets[i] = model.predict(state_t)[0]

            Q_sa = np.max(model.predict(state_tp1)[0])

            if game_over:
                targets[i, action_t] = reward_t
            else:
                targets[i, action_t] = reward_t + self.discount * Q_sa

        return inputs, targets


loss_list = []
reward_list = []


def train(env, memory: Memory, model, epochs, epsilon=.9, batch_size=10, timesteps=200):
    for i_episode in range(epochs):
        observation = env.reset()

        loss_on_epochs = 0
        reward_on_epochs = 0

        for t in range(timesteps):
            state_0 = np.array(observation, ndmin=2)

            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(model.predict(state_0)[0])

            observation, reward, done = env.step(action)
            reward_on_epochs += reward
            state_1 = np.array(observation, ndmin=2)

            memory.remember([state_0, action, reward, state_1], done)

            inputs, targets = memory.get_batch(model, max_batch_size=batch_size)

            loss = model.train_on_batch(inputs, targets)
            loss_on_epochs += loss

        loss_list.append(loss_on_epochs)
        reward_list.append(reward_on_epochs)

    env.reset()


def main():
    epsilon = .10  # probability of choosing a random action instead of using the model to decide
    max_memory = 200  # max number of experiences to be stored at once
    hidden_size = 100  # size of the hidden layers within the network
    batch_size = 20  # amount of experiences to sample into each batch for training
    discount = .95  # value of future reward vs. current reward
    learning_rate = .005  # the multiplicative rate at which the weights of the model are shifted
    timesteps = 10  # length of each game (for Cartpole, ideally set this to between 100-200)
    epochs = 1  # (Amount of games played)

    env = ClassifierEnv()

    feature_dims = len(env.reset())
    num_actions = env.action_space.n

    memory = Memory(max_memory=max_memory, discount=discount, env_dim=feature_dims)

    model = Sequential()
    model.add(Dense(hidden_size, input_shape=(feature_dims,), activation='relu'))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(num_actions))
    model.compile(Adam(learning_rate), "mse")

    train(
        env=env, memory=memory, model=model, epochs=epochs,
        epsilon=epsilon, batch_size=batch_size,
        timesteps=timesteps
    )

    model.save_weights(f'model.weights.h5')


if __name__ == '__main__':
    main()
