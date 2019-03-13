import random


class Agent:
    def __init__(self, environment):
        self.environment = environment

    def produce_state_and_get_action(self, last_date=None):
        state = self.environment.produce_state(self, last_date)
        if state is None:
            # case when next day's state is not available
            return None, None
        action = self.get_action(state)
        return state, action

    @staticmethod
    def get_epsilon_for_iteration(current_iteration, stable_iteration=1000000, initial_epsilon=1,
                                  end_epsilon=0.1):
        """
        decrease the epsilon linearly from 1 to 0.1 over the first million times, and fixed at 0.1 thereafter
        """
        epsilon = end_epsilon

        if current_iteration <= stable_iteration:
            decrease_per_epsilon = (initial_epsilon - end_epsilon) / stable_iteration
            epsilon = initial_epsilon - current_iteration * decrease_per_epsilon

        return epsilon

    def get_action(self, state):
        iteration = 1000000  # use static epsilon for now

        # Choose epsilon based on the iteration
        epsilon = self.get_epsilon_for_iteration(iteration, end_epsilon=0.5)

        # Choose the action
        if random.random() < epsilon:
            action = self.model.get_random_action()
        else:
            action = self.model.predict(state)

        print(self.__class__.__name__ + ": " + str(action))

        return action
