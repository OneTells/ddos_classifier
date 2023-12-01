from random import choice

from modules.learning.environment import ClassifierEnv


class RunnerEnv:

    def __init__(self):
        self.__env = ClassifierEnv()
        self.call_count = 0

    def __call__(self):
        self.call_count += 1

        is_done = False
        score = 0

        while not is_done:
            _, reward, is_done, _, _ = self.__env.step(action := choice(range(11)))
            score += reward

            print(f'Кол-во очков: {score} | Действие: {action}')

        print(f"Вызов: {self.call_count} | Кол-во очков: {score}")
        self.__env.reset()


def main():
    env = RunnerEnv()

    for _ in range(1):
        env()


if __name__ == "__main__":
    main()
