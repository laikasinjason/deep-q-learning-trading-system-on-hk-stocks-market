import random

import keras
import numpy as np
from keras import backend as keras_backend
from sklearn.preprocessing import OneHotEncoder


# Note: pass in_keras=False to use this function with raw numbers of numpy arrays for testing
def huber_loss(a, b, in_keras=True):
    error = a - b
    quadratic_term = error * error / 2
    linear_term = abs(error) - 1 / 2
    use_linear_term = (abs(error) > 1.0)
    if in_keras:
        # Keras won't let us multiply floats by booleans, so we explicitly cast the booleans to floats
        use_linear_term = keras_backend.cast(use_linear_term, 'float32')
    return use_linear_term * linear_term + (1 - use_linear_term) * quadratic_term


class Model:
    # tensorflow backend
    """
    Nh=Ns(α/(Ni+No))
    Ni = number of input neurons.
    No = number of output neurons.
    Ns = number of samples in training data set.
    α = an arbitrary scaling factor usually 2-10.
    """

    def __init__(self, n_actions, n_states):
        self.n_actions = n_actions
        self.n_states = n_states
        self.one_hot_encoder = OneHotEncoder(sparse=False)

    def copy_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def fit(self, state, reward, action_value):
        # Run one fast-forward to get the Q-values for all actions
        state = np.expand_dims(state, axis=0)
        target = self.model.predict(state)

        # Set the new Q values to target
        one_hot_action = self.value_map_to_action(action_value)

        target[one_hot_action.astype(bool)] = reward

        loss = self.model.fit(state, target, epochs=1, batch_size=1, verbose=0)

        return loss

    def predict(self, state):
        state = np.expand_dims(state, axis=0)
        action_pos = self.model.predict(state).argmax()

        action_value = self.action_map_to_value(action_pos)
        print("Predicted action: " + str(action_value))
        return action_value

    def get_random_action(self):
        action_value = random.choice(list(self.action_map.keys()))
        print("Random action: " + str(action_value))
        return action_value

    def action_map_to_value(self, search_action):
        for value, action in self.action_map.items():
            if (action == search_action).all():
                return value

    def value_map_to_action(self, value):
        value = self.action_map.get(value)
        action = self.one_hot_encoder.transform([[value]])
        return action


class OrderModel(Model):
    # action map the value (percentage of ma(5)) to the output form from the model
    action_map = {-3: 0,
                  -2: 1,
                  -1: 2,
                  0: 3,
                  1: 4,
                  2: 5,
                  3: 6}

    def __init__(self, n_actions, n_states):
        super().__init__(n_actions, n_states)
        # DQN model
        self.model = self.__create_model()

    def __create_model(self, alpha=0.00025):
        # States for Order Agent (-3% to +3%): { -3, -2, -1, 0, 1, 2, 3 }

        model_input = keras.layers.Input((self.n_states,), name='inputs')
        layer_1 = keras.layers.Dense(256, activation='relu')(model_input)
        layer_2 = keras.layers.Dense(128, activation='relu')(layer_1)
        layer_3 = keras.layers.Dense(64, activation='relu')(layer_2)
        output = keras.layers.Dense(self.n_actions)(layer_3)

        model = keras.models.Model(input=[model_input], output=output)
        optimizer = keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
        model.compile(optimizer, loss=huber_loss)

        self.one_hot_encoder.fit(np.array([i for i in range(self.n_actions)]).reshape(-1, 1))

        print("Created model with action " + str(self.n_actions) + ", state " + str(self.n_states))

        return model


class SignalModel(Model):
    action_map = {False: 0,
                  True: 1}

    def __init__(self, n_actions, n_states):
        super().__init__(n_actions, n_states)
        # DQN model
        self.model = self.__create_model()

    def __create_model(self, alpha=0.00025):
        model_input = keras.layers.Input((self.n_states,), name='inputs')
        layer_1 = keras.layers.Dense(256, activation='relu')(model_input)
        layer_2 = keras.layers.Dense(128, activation='relu')(layer_1)
        layer_3 = keras.layers.Dense(64, activation='relu')(layer_2)
        output = keras.layers.Dense(self.n_actions)(layer_3)

        model = keras.models.Model(input=[model_input], output=output)
        optimizer = keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
        model.compile(optimizer, loss=huber_loss)

        self.one_hot_encoder.fit(np.array([i for i in range(self.n_actions)]).reshape(-1, 1))

        print("Created model with action " + str(self.n_actions) + ", state " + str(self.n_states))

        return model


class SellSignalModel(SignalModel):
    gamma = 0.99

    def __init__(self, n_actions, n_states):
        super().__init__(n_actions, n_states)
        # target DQN model for smoothing the learning process
        self.target_model = self.__create_model()

    # override the fit method, since sell signal agent has diff training algo
    def fit(self, state, reward, action_value, next_state):
        # Run one fast-forward to get the Q-values for all actions
        state = np.expand_dims(state, axis=0)
        target = self.model.predict(state)

        next_state = np.expand_dims(next_state, axis=0)
        next_Q_values = self.target_model.predict(next_state)
        new_Q_values = reward + self.gamma * np.max(next_Q_values, axis=1)

        # Set the new Q values to target
        one_hot_action = self.value_map_to_action(action_value)

        target[one_hot_action.astype(bool)] = new_Q_values

        loss = self.model.fit(state, target, epochs=1, batch_size=1, verbose=0)

        return loss
        
    def save_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
