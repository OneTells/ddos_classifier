from keras.models import Sequential
from keras.src.layers import Dense
from keras.src.optimizers import Adam

from modules.learning.agent import Agent
from modules.learning.environment import ClassifierEnv
from modules.learning.memory import Memory


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
