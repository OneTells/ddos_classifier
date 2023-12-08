import numpy as np
from keras import Sequential
from keras.optimizer_v2.adam import Adam
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy
from tensorflow.keras.layers import Dense, Flatten

from modules.learning.environment import ClassifierEnv


class RunnerEnv:

    @staticmethod
    def create_model(states: int, actions: int) -> Sequential:
        model = Sequential()

        model.add(Flatten(input_shape=(1, states)))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(actions, activation="linear"))

        return model

    @classmethod
    def run(cls):
        env = ClassifierEnv()

        states = env.observation_space.shape[0]
        actions = env.action_space.n

        agent = DQNAgent(
            model=cls.create_model(states, actions),
            memory=SequentialMemory(limit=50000, window_length=1),
            policy=BoltzmannQPolicy(),
            nb_actions=actions,
            nb_steps_warmup=10,
            target_model_update=0.01
        )

        agent.compile(Adam(lr=0.001), metrics=["mae"])
        agent.fit(env, nb_steps=100000, visualize=False, verbose=1)

        results = agent.test(env, nb_episodes=10, visualize=True)
        print(np.mean(results.history["episode_reward"]))

        env.close()


def main():
    RunnerEnv.run()


if __name__ == "__main__":
    main()
