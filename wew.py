import numpy as np
from keras.models import Sequential
from keras.src.layers import Dense
from keras.src.optimizers import Adam

from modules.learning.environment import ClassifierEnv


def test(model, games=2, timesteps=10, output=1, render=0):
    """
    Test our model and output results.

    Inputs:
    model: model to be tested
    games: amount of games to play
    timesteps: length of each game
    output: enable/disable print output
    render: enable/disable rendering
    """

    Win_count = 0

    env = ClassifierEnv()
    for game in range(games):
        observation = env.reset()
        total_reward = 0

        for t in range(timesteps):
            state = np.array(observation, ndmin=2)
            action = np.argmax(model.predict(state)[0])
            observation, reward, done = env.step(action)
            total_reward += reward

            if done or t >= timesteps - 1:
                if t >= timesteps - 1:
                    Win_count += 1
                break

        print(total_reward / timesteps)

    if output == 1:
        print("Test results: {}/{} games won.".format(Win_count, games))


def main():
    epsilon = .10  # probability of choosing a random action instead of using the model to decide
    max_memory = 200  # max number of experiences to be stored at once
    hidden_size = 100  # size of the hidden layers within the network
    batch_size = 20  # amount of experiences to sample into each batch for training
    discount = .95  # value of future reward vs. current reward
    learning_rate = .005  # the multiplicative rate at which the weights of the model are shifted
    timesteps = 20  # length of each game (for Cartpole, ideally set this to between 100-200)
    epochs = 1  # (Amount of games played)
    set_size = 10  # rate at which games are rendered and progress is reported
    env = ClassifierEnv()
    feature_dims = len(env.reset())
    num_actions = env.action_space.n
    model = Sequential()
    model.add(Dense(hidden_size, input_shape=(feature_dims,), activation='relu'))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(num_actions))
    model.compile(Adam(learning_rate), "mse")
    model.load_weights(f'model.weights.h5')
    test(model, games=10, render=1, output=1, )


if __name__ == '__main__':
    main()
