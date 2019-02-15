import keras

from sklearn.preprocessing import OneHotEncoder

from keras import backend as K
# Note: pass in_keras=False to use this function with raw numbers of numpy arrays for testing
def huber_loss(a, b, in_keras=True):
    error = a - b
    quadratic_term = error*error / 2
    linear_term = abs(error) - 1/2
    use_linear_term = (abs(error) > 1.0)
    if in_keras:
        # Keras won't let us multiply floats by booleans, so we explicitly cast the booleans to floats
        use_linear_term = K.cast(use_linear_term, 'float32')
    return use_linear_term * linear_term + (1-use_linear_term) * quadratic_term
    
class Model:
    # tensorflow backend
    '''
    Nh=Ns(α/(Ni+No))
    Ni = number of input neurons.
    No = number of output neurons.
    Ns = number of samples in training data set.
    α = an arbitrary scaling factor usually 2-10.
    '''
    @staticmethod
    def __create_model(n_actions, n_states, alpha=0.00025):

        # # Assuming that the input frames aimgre still encoded from 0 to 255. Transforming to [0, 1].
        # #         normalized = keras.layers.Lambda(lambda x: x / 255.0)(frames_input)
        
        input = keras.layers.Input(n_states, name='inputs')
        layer_1 = keras.layers.Dense(256, activation='relu')(input)
        layer_2 = keras.layers.Dense(128, activation='relu')(layer_1)
        layer_3 = keras.layers.Dense(64, activation='relu')(layer_2)
        output = keras.layers.Dense(n_actions)(layer_3)

        model = keras.models.Model(input=[input], output=output)
        optimizer = keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
        model.compile(optimizer, loss=huber_loss)

        self.one_hot_encoder.fit(np.array([i for i in range(n_actions)]))
        
        print("Created model with action "+ n_actions + ", state " + n_states)
        
        return model
        

    def __init__(self, n_actions, n_states):
        self.n_actions = n_actions
        self.n_states = n_states
        self.one_hot_encoder = OneHotEncoder(sparse=False)

        self.model = self.__create_model(self.n_actions, self.n_states)
        self.target_model = self.__create_model(self.n_actions, self.n_states)

    def copy_model(self):
        self.target_model.set_weights(self.model.get_weights())
        
    def fit(self, states, reward, action):
        # Run one fast-forward to get the Q-values for all actions
        target = self.model.predict(states)
        next_Q_values = reward
        
        # Set the new Q values to target
        one_hot_action = self.one_hot_encoder.transform(action)
        target[one_hot_action.astype(bool)] = new_Q_values
        
        loss = model.fit(states, target, epochs=1, batch_size=len(states), verbose=0)
        
        return loss

    def predict(self, state):
        return self.model.predict(state)