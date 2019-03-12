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

    def fit(self, states, reward, action):
        # Run one fast-forward to get the Q-values for all actions
        target = self.model.predict(states)
        new_Q_values = reward

        # Set the new Q values to target
        one_hot_action = self.one_hot_encoder.transform(action)
        target[one_hot_action.astype(bool)] = new_Q_values

        loss = model.fit(states, target, epochs=1, batch_size=1, verbose=0)

        return loss

    def predict(self, state):
        return self.model.predict(state)
        
        
        
class OrderModel(Model):
    def __init__(self, n_actions, n_states):
        super().__init__(n_actions, n_states)
        self.model = self.__create_model()
        
        # action map the value (percentage of ma(5)) to the output form from the model
        self.__action_map = {-3 : [1., 0., 0., 0., 0., 0., 0.],
                             -2 : [0., 1., 0., 0., 0., 0., 0.],
                             -1 : [0., 0., 1., 0., 0., 0., 0.],
                              0 : [0., 0., 0., 1., 0., 0., 0.],
                              1 : [0., 0., 0., 0., 1., 0., 0.],
                              2 : [0., 0., 0., 0., 0., 1., 0.],
                              3 : [0., 0., 0., 0., 0., 0., 1.]}
        # self.target_model = self.__create_model()
        
    def __create_model(self, alpha=0.00025):
        # States for Order Agent (-3% to +3%): { -3, -2, -1, 0, 1, 2, 3 }
        
        input = keras.layers.Input((self.n_states,), name='inputs')
        layer_1 = keras.layers.Dense(256, activation='relu')(input)
        layer_2 = keras.layers.Dense(128, activation='relu')(layer_1)
        layer_3 = keras.layers.Dense(64, activation='relu')(layer_2)
        output = keras.layers.Dense(self.n_actions)(layer_3)

        model = keras.models.Model(input=[input], output=output)
        optimizer = keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
        model.compile(optimizer, loss=huber_loss)

        self.one_hot_encoder.fit(np.array([i for i in range(self.n_actions)]).reshape(-1, 1))

        print("Created model with action " + str(self.n_actions) + ", state " + str(self.n_states))

        return model
        
    def action_map_to_value(search_action):
        for value, action in self.__action_map.items():
            if action == search_action:
                return value
    
    def value_map_to_action(value):
        return self.__action_map.get(value)
        
class SignalModel(Model):
    def __init__(self, n_actions, n_states):
        super().__init__(n_actions, n_states)
        self.model = self.__create_model()
        # self.target_model = self.__create_model()
        
    def __create_model(self, alpha=0.00025):
        
        input = keras.layers.Input((self.n_states,), name='inputs')
        layer_1 = keras.layers.Dense(256, activation='relu')(input)
        layer_2 = keras.layers.Dense(128, activation='relu')(layer_1)
        layer_3 = keras.layers.Dense(64, activation='relu')(layer_2)
        output = keras.layers.Dense(self.n_actions)(layer_3)

        model = keras.models.Model(input=[input], output=output)
        optimizer = keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
        model.compile(optimizer, loss=huber_loss)

        self.one_hot_encoder.fit(np.array([i for i in range(self.n_actions)]).reshape(-1, 1))

        print("Created model with action " + str(self.n_actions) + ", state " + str(self.n_states))

        return model

        
class SellSignalModel(SignalModel):
    def __init__(self, n_actions, n_states):
        super().__init__(n_actions, n_states)
        
    # override the fit method, since sell signal agent has diff training algo
    def fit(self, states, reward, action):
        # Run one fast-forward to get the Q-values for all actions
        target = self.model.predict(states)
        next_Q_values = target_model.predict(next_states)
        new_Q_values = rewards + gamma * np.max(next_Q_values, axis=1)

        # Set the new Q values to target
        one_hot_action = self.one_hot_encoder.transform(action)
        target[one_hot_action.astype(bool)] = new_Q_values

        loss = model.fit(states, target, epochs=1, batch_size=1, verbose=0)

        return loss