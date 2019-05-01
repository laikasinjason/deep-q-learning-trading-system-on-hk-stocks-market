import random

import keras
import numpy as np
import tensorflow as tf  # Deep Learning library
from keras import backend as keras_backend
from sklearn.preprocessing import OneHotEncoder

from prioritized_exp_replay import Memory


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


class DDDQNNet:
    def __init__(self, state_size, action_size, learning_rate, sess, name):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.name = name

        # We use tf.variable_scope here to know which network we're using (DQN or target_net)
        # it will be useful when we will update our w- parameters (by copy the DQN parameters)
        with sess.graph.as_default():
            with tf.variable_scope(self.name):
                # We create the placeholders
                # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
                # [None, 100, 120, 4]
                self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")

                #
                self.ISWeights_ = tf.placeholder(tf.float32, [None, 1], name='IS_weights')

                self.actions_ = tf.placeholder(tf.float32, [None, action_size], name="actions_")

                # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
                self.target_Q = tf.placeholder(tf.float32, [None], name="target")

                # Input
                self.dense1 = tf.layers.dense(inputs=self.inputs_,
                                              units=512,
                                              activation=tf.nn.elu,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                              name="dense1")
                self.dense2 = tf.layers.dense(inputs=self.dense1,
                                              units=256,
                                              activation=tf.nn.elu,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                              name="dense2")

                # Here we separate into two streams
                # The one that calculate V(s)
                self.value_fc = tf.layers.dense(inputs=self.dense2,
                                                units=128,
                                                activation=tf.nn.elu,
                                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                name="value_fc")

                self.value = tf.layers.dense(inputs=self.value_fc,
                                             units=1,
                                             activation=None,
                                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                             name="value")

                # The one that calculate A(s,a)
                self.advantage_fc = tf.layers.dense(inputs=self.dense2,
                                                    units=128,
                                                    activation=tf.nn.elu,
                                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                    name="advantage_fc")

                self.advantage = tf.layers.dense(inputs=self.advantage_fc,
                                                 units=self.action_size,
                                                 activation=None,
                                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                 name="advantages")

                # Aggregating layer
                # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a'))
                self.output = self.value + tf.subtract(self.advantage,
                                                       tf.reduce_mean(self.advantage, axis=1, keepdims=True))

                # Q is our predicted Q value.
                self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)

                # The loss is modified because of PER
                self.absolute_errors = tf.abs(self.target_Q - self.Q)  # for updating Sumtree

                self.loss = tf.reduce_mean(self.ISWeights_ * tf.squared_difference(self.target_Q, self.Q))

                self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)


class Model:
    # tensorflow backend
    """
    Nh=Ns(α/(Ni+No))
    Ni = number of input neurons.
    No = number of output neurons.
    Ns = number of samples in training data set.
    α = an arbitrary scaling factor usually 2-10.
    """
    memory_size = 10000
    learning_rate = 0.00025  # Alpha (aka learning rate)
    sess = tf.Session()
    saver = None
    writer = None

    def __init__(self, n_actions, n_states, batch_size):
        self.n_actions = n_actions
        self.n_states = n_states
        self.batch_size = batch_size
        self.memory = Memory(self.memory_size)
        self.one_hot_encoder = OneHotEncoder(sparse=False)

    @classmethod
    def init(cls):
        with cls.sess.graph.as_default():
            cls.sess.run(tf.global_variables_initializer())
            cls.saver = tf.train.Saver()
            # Setup TensorBoard Writer
            cls.writer = tf.summary.FileWriter("/home/laikasin93/python/tensorboard/dddqn/1")
            cls.writer.add_graph(cls.sess.graph)

    def copy_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def fit(self, state, reward, action_value, next_state=None):
        self.store_experience(state, reward, action_value)

        if self.memory.is_full():
            self.fit_model(self.batch_size)

            """
            tree_idx, batch, ISWeights = self.memory.sample(self.batch_size)
            
            states = np.array([each[0][0] for each in batch])
            actions = np.array([each[0][1] for each in batch])
            rewards = np.array([each[0][2] for each in batch]) 
                
            # Run one fast-forward to get the Q-values for all actions
            states = np.expand_dims(states, axis=0)
            targets = self.model.predict(states)

            # Set the new Q values to target
            one_hot_actions = self.value_map_to_action(actions)

            targets[one_hot_actions.astype(bool)] = rewards

            loss = self.model.fit(states, targets, epochs=1, batch_size=len(rewards), verbose=0)
            """

    def store_experience(self, state, reward, action_value, next_state=None):
        # prioritized replay
        transition = state, action_value, reward
        self.memory.store(transition)  # have high priority for newly arrived transition

    def predict(self, state):
        # state = np.expand_dims(state, axis=0)
        # action_pos = self.model.predict(state).argmax()

        Qs = self.sess.run(self.model.output, feed_dict={self.model.inputs_: state.reshape((1, *state.shape))})
        # Take the biggest Q value (= the best action)
        action_pos = np.argmax(Qs)

        action_value = self.action_map_to_value(action_pos)
        print("Predicted action: " + str(action_value))
        return action_value

    def get_random_action(self):
        action_value = random.choice(list(self.action_map.keys()))
        # print("Random action: " + str(action_value))
        return action_value

    def action_map_to_value(self, search_action):
        for value, action in self.action_map.items():
            if (action == search_action).all():
                return value

    def value_map_to_action(self, value):
        value = list(map(lambda x: self.action_map.get(x), value))
        action = self.one_hot_encoder.transform([[x] for x in value])
        return action

    def fit_model(self, batch_size):
        # LEARNING PART
        # Obtain random mini-batch from memory
        tree_idx, batch, ISWeights_mb = self.memory.sample(self.batch_size)

        states_mb = np.array([each[0][0] for each in batch])
        actions_mb = np.array([each[0][1] for each in batch])
        rewards_mb = np.array([each[0][2] for each in batch])

        target_Qs_batch = []

        actions_mb = self.value_map_to_action(actions_mb)

        # Set Q_target = r
        for i in range(0, len(batch)):
            # Reward as target
            target_Qs_batch.append(rewards_mb[i])

        targets_mb = np.array([each for each in target_Qs_batch])

        _, loss, absolute_errors = self.sess.run([self.model.optimizer, self.model.loss, self.model.absolute_errors],
                                                 feed_dict={self.model.inputs_: states_mb,
                                                            self.model.target_Q: targets_mb,
                                                            self.model.actions_: actions_mb,
                                                            self.model.ISWeights_: ISWeights_mb})

        # Update priority
        self.memory.batch_update(tree_idx, absolute_errors)


        # Write TF Summaries
        summary = sess.run(write_op, feed_dict={DQNetwork.inputs_: states_mb,
                                                DQNetwork.target_Q: targets_mb,
                                                DQNetwork.actions_: actions_mb,
                                                DQNetwork.ISWeights_: ISWeights_mb})
        writer.add_summary(summary, episode)
        writer.flush()


class OrderModel(Model):
    # action map the value (percentage of ma(5)) to the output form from the model
    action_map = {-3: 0,
                  -2: 1,
                  -1: 2,
                  0: 3,
                  1: 4,
                  2: 5,
                  3: 6}

    def __init__(self, n_actions, n_states, batch_size, name):
        super().__init__(n_actions, n_states, batch_size)
        # DQN model
        # self.model = self._create_model()
        # Instantiate the DQNetwork
        self.model = DDDQNNet(n_states, n_actions, self.learning_rate, self.sess, name=(name + "DQNetwork"))

    def _create_model(self, alpha=0.00025):
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

    def __init__(self, n_actions, n_states, batch_size, name):
        super().__init__(n_actions, n_states, batch_size)
        # DQN model
        # self.model = self._create_model()
        # Instantiate the DQNetwork
        self.model = DDDQNNet(n_states, n_actions, self.learning_rate, self.sess, name=(name + "DQNetwork"))

    def _create_model(self, alpha=0.00025):
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
    gamma = 0.99  # Discounting rate

    def __init__(self, n_actions, n_states, batch_size, name):
        super().__init__(n_actions, n_states, batch_size, name)
        # target DQN model for smoothing the learning process
        # self.target_model = super()._create_model()
        # Instantiate the target network
        self.target_model = DDDQNNet(n_states, n_actions, self.learning_rate,self.sess, name=(name + "TargetNetwork"))

    # override the fit method, since sell signal agent has diff training algo
    def fit(self, state, reward, action_value, next_state=None):
        self.store_experience(state, reward, action_value, next_state)

        if self.memory.is_full():
            self.fit_model(self.batch_size)

        """
            tree_idx, batch, ISWeights = self.memory.sample(self.batch_size)
            
            states = np.array([each[0][0] for each in batch])
            actions = np.array([each[0][1] for each in batch])
            rewards = np.array([each[0][2] for each in batch]) 
                
            # Run one fast-forward to get the Q-values for all actions
            states = np.expand_dims(states, axis=0)
            targets = self.model.predict(states)

            # Set the new Q values to target
            one_hot_actions = self.value_map_to_action(actions)

            targets[one_hot_actions.astype(bool)] = rewards

            loss = self.model.fit(states, targets, epochs=1, batch_size=len(rewards), verbose=0)
         
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
        """

    def store_experience(self, state, reward, action_value, next_state=None):
        # prioritized replay
        transition = state, action_value, reward, next_state
        self.memory.store(transition)  # have high priority for newly arrived transition

    def save_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # Copy the parameters of DQN to Target_network
    def update_target_graph(self):

        # Get the parameters of our DQNNetwork
        from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, str(self.__class__.__name__) + "DQNetwork")

        # Get the parameters of our Target_network
        to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, str(self.__class__.__name__) + "TargetNetwork")

        op_holder = []

        # Update our target_network parameters with DQNNetwork parameters
        for from_var, to_var in zip(from_vars, to_vars):
            op_holder.append(to_var.assign(from_var))

        self.sess.run(op_holder)
        print("Target Model updated")

    def fit_model(self, batch_size):
        # LEARNING PART
        # Obtain random mini-batch from memory
        tree_idx, batch, ISWeights_mb = self.memory.sample(self.batch_size)

        states_mb = np.array([each[0][0] for each in batch])
        actions_mb = np.array([each[0][1] for each in batch])
        rewards_mb = np.array([each[0][2] for each in batch])
        next_states_mb = np.array([each[0][3] for each in batch])

        target_Qs_batch = []

        actions_mb = self.value_map_to_action(actions_mb)

        # DOUBLE DQN Logic
        # Use DQNNetwork to select the action to take at next_state (a') (action with the highest Q-value)
        # Use TargetNetwork to calculate the Q_val of Q(s',a')

        # Get Q values for next_state 
        q_next_state = self.sess.run(self.model.output, feed_dict={self.model.inputs_: next_states_mb})

        # Calculate Qtarget for all actions that state
        q_target_next_state = self.sess.run(self.target_model.output,
                                            feed_dict={self.target_model.inputs_: next_states_mb})

        # Q_target = r + gamma * Qtarget(s',a') 
        for i in range(0, len(batch)):
            # We got a'
            action = np.argmax(q_next_state[i])

            target = rewards_mb[i] + self.gamma * q_target_next_state[i][action]
            target_Qs_batch.append(target)

        targets_mb = np.array([each for each in target_Qs_batch])

        _, loss, absolute_errors = self.sess.run([self.model.optimizer, self.model.loss, self.model.absolute_errors],
                                                 feed_dict={self.model.inputs_: states_mb,
                                                            self.model.target_Q: targets_mb,
                                                            self.model.actions_: actions_mb,
                                                            self.model.ISWeights_: ISWeights_mb})

        # Update priority
        self.memory.batch_update(tree_idx, absolute_errors)

        # Write TF Summaries
        # summary = self.sess.run(write_op, feed_dict={DQNetwork.inputs_: states_mb,
        #                                    DQNetwork.target_Q: targets_mb,
        #                                    DQNetwork.actions_: actions_mb,
        #                               DQNetwork.ISWeights_: ISWeights_mb})
        # writer.add_summary(summary, episode)
        # writer.flush()
