import keras


class Model:
    # tensorflow backend
    @staticmethod
    def __create_model(n_actions, alpha=0.00025):
        # # tensorflow backend
        # ATARI_SHAPE = (84, 84, 4)
        #
        # # With the functional API we need to define the inputs.
        # frames_input = keras.layers.Input(ATARI_SHAPE, name='frames')
        #
        # # Assuming that the input frames aimgre still encoded from 0 to 255. Transforming to [0, 1].
        # #         normalized = keras.layers.Lambda(lambda x: x / 255.0)(frames_input)
        #
        # # "The first hidden layer convolves 16 8×8 filters with stride 4 with
        # #  the input image and applies a rectifier nonlinearity."
        # conv_1 = keras.layers.convolutional.Convolution2D(16, 8, 8, subsample=(4, 4), activation='relu')(frames_input)
        # # "The second hidden layer convolves 32 4×4 filters with stride 2, again followed by a rectifier nonlinearity."
        # conv_2 = keras.layers.convolutional.Convolution2D(32, 4, 4, subsample=(2, 2), activation='relu')(conv_1)
        # # Flattening the second convolutional layer.
        # conv_flattened = keras.layers.core.Flatten()(conv_2)
        # # "The final hidden layer is fully-connected and consists of 256 rectifier units."
        # hidden = keras.layers.Dense(256, activation='relu')(conv_flattened)
        # # "The output layer is a fully-connected linear layer with a single output for each valid action."
        # output = keras.layers.Dense(n_actions)(hidden)
        #
        # model = keras.models.Model(input=[frames_input], output=output)
        # optimizer = keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
        # model.compile(optimizer, loss=huber_loss)

        # return model
        pass

    def __init__(self, n_actions, n_states):
        self.n_actions = n_actions
        self.n_states = n_states

        self.model = self.__create_model(self.n_actions, self.n_states)
        self.target_model = self.__create_model(self.n_actions, self.n_states)

    def copy_model(self):
        self.target_model.set_weights(self.model.get_weights())
