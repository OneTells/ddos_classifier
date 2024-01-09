import os
from matplotlib import pyplot as plt

from keras.models import Sequential
from keras.src.layers import Dense
from keras.src.optimizers import Adam
from sklearn.metrics import classification_report

from modules.learning.agent import Agent
from modules.learning.environment import ClassifierEnv
from modules.learning.memory import Memory


def create_model(hidden_size: float, feature_dims: int, num_actions: int, learning_rate: float) -> Sequential:
    model = Sequential()

    model.add(Dense(hidden_size, input_shape=(feature_dims,), activation='relu'))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(num_actions))
    model.compile(Adam(learning_rate), "mse")

    return model


def main():
    epsilon = 0.10  # probability of choosing a random action instead of using the model to decide
    hidden_size = 5  # size of the hidden layers within the network
    discount = 0.95  # value of future reward vs. current reward
    learning_rate = 0.005  # the multiplicative rate at which the weights of the model are shifted

    max_memory = 10  # max number of experiences to be stored at once
    batch_size = 5  # amount of experiences to sample into each batch for training

    epochs = 5  # (Amount of games played)
    time_steps = 10  # length of each game (for Cartpole, ideally set this to between 100-200)

    dataset_path = f'{os.getcwd()}/data/super_optimize_two_dataset.bz2'

    env = ClassifierEnv(dataset_path)

    feature_dims = len(env.reset())
    num_actions = env.action_space.n

    memory = Memory(feature_dims, max_memory, discount)

    model = create_model(hidden_size, feature_dims, num_actions, learning_rate)

    agent = Agent(env, memory, model, epochs, epsilon, batch_size, time_steps)
    agent.train()

    print(agent.filters_history)
    # model.save_weights(f'model.weights.h5')

    # model.load_weights(f'model.weights.h5')
    # agent.test()

    plt.plot(agent.loss_list)

    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()

    plt.plot(agent.reward_list)

    print(agent.reward_list)
    print(agent.loss_list)

    plt.title('model reward')
    plt.ylabel('reward')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()

    result = classification_report(env.report_y_true, env.report_y_answer)
    print(result)


if __name__ == '__main__':
    main()
