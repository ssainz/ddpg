import tensorflow as tf
import numpy as np
import tflearn


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.
    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        # Create the critic network
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
            + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, self.a_dim])

        # Define loss and optimization Op
        self.loss2 = tflearn.mean_square(self.predicted_q_value, self.out)

        reduced_sum = tf.reduce_sum(tf.pow(self.predicted_q_value - self.out,2), axis=1) / 3
        self.loss = tf.reduce_mean(reduced_sum)

        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])

        w1 = tflearn.fully_connected(inputs, 300)
        #w1 = tflearn.layers.normalization.batch_normalization(w1)
        w1 = tflearn.activations.relu(w1)

        h1 = tflearn.fully_connected(w1, 600)
        #h1 = tflearn.layers.normalization.batch_normalization(h1)
        h1 = tflearn.activations.linear(h1)

        a1 = tflearn.fully_connected(action, 600)
        #a1 = tflearn.layers.normalization.batch_normalization(a1)
        a1 = tflearn.activations.linear(a1)

        h2 = tflearn.layers.merge_ops.merge([h1,a1], mode='elemwise_sum')

        h3 = tflearn.fully_connected(h2, 600)
        #h3 = tflearn.layers.normalization.batch_normalization(h3)
        h3 = tflearn.activations.relu(h3)

        V = tflearn.fully_connected(h3, self.a_dim)
        #V = tflearn.layers.normalization.batch_normalization(V)
        V = tflearn.activations.linear(V)

        return inputs, action, V

        # net = tflearn.fully_connected(inputs, 400)
        # net = tflearn.layers.normalization.batch_normalization(net)
        # net = tflearn.activations.relu(net)
        #
        # # Add the action tensor in the 2nd hidden layer
        # # Use two temp layers to get the corresponding weights and biases
        # t1 = tflearn.fully_connected(net, 300)
        # t2 = tflearn.fully_connected(action, 300)
        #
        # net = tflearn.activation(
        #     tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')
        #
        # # linear layer connected to 1 output representing Q(s,a)
        # # Weights are init to Uniform[-3e-3, 3e-3]
        # w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        # out = tflearn.fully_connected(net, self.a_dim, weights_init=w_init)
        # return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize, self.loss, self.loss2], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

