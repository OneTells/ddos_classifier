import os

from keras.models import Sequential
from keras.src.layers import Dense
from keras.src.optimizers import Adam
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report

from modules.learning.agent import Agent
from modules.learning.environment import ClassifierEnv
from modules.learning.filter import Filter
from modules.learning.memory import Memory


def create_model(hidden_size: float, feature_dims: int, num_actions: int, learning_rate: float) -> Sequential:
    model = Sequential()

    model.add(Dense(hidden_size, input_shape=(feature_dims,), activation='relu'))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(num_actions))
    model.compile(Adam(learning_rate), "mse")

    return model


class Main:
    epsilon = 0.10  # probability of choosing a random action instead of using the model to decide
    hidden_size = 5  # size of the hidden layers within the network
    discount = 0.95  # value of future reward vs. current reward
    learning_rate = 0.005  # the multiplicative rate at which the weights of the model are shifted

    max_memory = 10  # max number of experiences to be stored at once
    batch_size = 5  # amount of experiences to sample into each batch for training

    @staticmethod
    def __draw_loss(agent: Agent) -> None:
        plt.plot(agent.loss_list)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.show()

    @staticmethod
    def __draw_reward(agent: Agent) -> None:
        plt.plot(agent.reward_list)
        plt.title('model reward')
        plt.ylabel('reward')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.show()

    @classmethod
    def __train(cls, epochs: int, time_steps: int) -> tuple[Agent, tuple[Filter, ...]]:
        dataset_path = f'{os.getcwd()}/data/super_optimize_two_dataset.bz2'
        env = ClassifierEnv(dataset_path)

        feature_dims = len(env.reset())
        num_actions = env.action_space.n

        model = create_model(cls.hidden_size, feature_dims, num_actions, cls.learning_rate)
        agent = Agent(model)

        memory = Memory(feature_dims, cls.max_memory, cls.discount)
        agent.train(env, memory, epochs, cls.epsilon, cls.batch_size, time_steps)

        print(agent.filters_history)

        cls.__draw_loss(agent)
        cls.__draw_reward(agent)

        return agent, env.filters

    @classmethod
    def __test(cls, agent: Agent, filters: tuple[Filter, ...]) -> None:
        dataset_path = f'{os.getcwd()}/data/super_optimize_one_dataset.bz2'
        env = ClassifierEnv(dataset_path)
        env.filters = filters

        agent.test(env, 3, 50)

        result = classification_report(env.report_y_true, env.report_y_answer)
        print(result)

        cls.__draw_reward(agent)

    @classmethod
    def run(cls) -> None:
        agent, last_filters = cls.__train(5, 10)
        print(f'{last_filters=}')
        cls.__test(agent, last_filters)


if __name__ == '__main__':
    Main.run()
