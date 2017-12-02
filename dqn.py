import state
import math
import numpy as np
import os
import tensorflow as tf

gamma = .99


class DeepQNetwork:
    def __init__(self, sess, numActions, baseDir, args):

        self.numActions = numActions
        self.baseDir = baseDir
        self.saveModelFrequency = args["save_model_freq"]
        self.targetModelUpdateFrequency = args["target_model_update_freq"]
        self.normalizeWeights = args["normalize_weights"]

        self.staleSess = None

        tf.set_random_seed(123456)

        # self.sess = tf.Session()
        self.sess = sess

        assert (len(tf.global_variables()) == 0), "Expected zero variables"
        self.x, self.y = self.buildNetwork('policy', True, numActions)
        assert (len(tf.trainable_variables()) == 10), "Expected 10 trainable_variables"
        assert (len(tf.global_variables()) == 10), "Expected 10 total variables"
        self.x_target, self.y_target = self.buildNetwork('target', False, numActions)
        assert (len(tf.trainable_variables()) == 10), "Expected 10 trainable_variables"
        assert (len(tf.global_variables()) == 20), "Expected 20 total variables"

        # build the variable copy ops
        self.update_target = []
        trainable_variables = tf.trainable_variables()
        all_variables = tf.global_variables()
        for i in range(0, len(trainable_variables)):
            # Update operation : update Q_target = Q.
            self.update_target.append(all_variables[len(trainable_variables) + i].assign(trainable_variables[i]))

        self.a = tf.placeholder(tf.float32, shape=[None, numActions])
        print('a %s' % (self.a.get_shape()))
        self.y_ = tf.placeholder(tf.float32, [None])
        print('y_ %s' % (self.y_.get_shape()))

        self.y_a = tf.reduce_sum(tf.multiply(self.y, self.a), reduction_indices=1)
        print('y_a %s' % (self.y_a.get_shape()))

        # difference = tf.abs(self.y_a - self.y_)
        # quadratic_part = tf.clip_by_value(difference, 0.0, 1.0)
        # linear_part = difference - quadratic_part
        # errors = (0.5 * tf.square(quadratic_part)) + linear_part
        # self.loss = tf.reduce_sum(errors)
        # self.loss = tf.reduce_mean(tf.square(tf.clip_by_value(self.y_a - self.y_, -1.0, 1.0)))
        difference = tf.abs(self.y_a - self.y_)
        quadratic_part = tf.clip_by_value(difference, 0.0, 1.0)
        linear_part = difference - quadratic_part
        errors = (0.5 * tf.square(quadratic_part)) + linear_part
        self.loss = tf.reduce_mean(errors)

        # (??) learning rate
        # Note tried gradient clipping with rmsprop with this particular loss function and it seemed to suck
        # Perhaps I didn't run it long enough
        # optimizer = GradientClippingOptimizer(tf.train.RMSPropOptimizer(args.learning_rate, decay=.95, epsilon=.01))
        #optimizer = tf.train.RMSPropOptimizer(args["learning_rate"], decay=.95, epsilon=.01)
        #self.train_step = optimizer.minimize(self.loss)


        opt = tf.train.AdamOptimizer(learning_rate=args["learning_rate"])
        # Compute the gradients for a list of variables.
        grads_and_vars = opt.compute_gradients(self.loss)

        # grads_and_vars is a list of tuples (gradient, variable).  Do whatever you
        # need to the 'gradient' part, for example cap them, etc.
        capped_grads_and_vars = []
        for grad, var in grads_and_vars:
            if grad is not None:
                grad_ = tf.clip_by_value(grad, -1., 1.)
            else:
                grad_ = tf.zeros_like(var)
            capped_grads_and_vars.append((grad_, var))
        self.train_step = opt.apply_gradients(capped_grads_and_vars)


        self.saver = tf.train.Saver(max_to_keep=25)

        # Initialize variables
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.update_target)  # is this necessary?

        self.summary_writer = tf.summary.FileWriter(self.baseDir + '/tensorboard', self.sess.graph)

        if args["model"] is not None:
            print('Loading from model file %s' % (args["model"]))
            self.saver.restore(self.sess, args["model"])

    def buildNetwork(self, name, trainable, numActions):

        print("Building network for %s trainable=%s" % (name, trainable))

        # First layer takes a screen, and shrinks by 2x
        x = tf.placeholder(tf.uint8, shape=[None, 128, 128, 4], name="screens")
        print(x)

        x_normalized = tf.to_float(x) / 255.0
        print(x_normalized)

        # Second layer convolves 32 8x8 filters with stride 4 with relu
        # ( ( 128 - 8) / 4 ) + 1 = 31. Final is: 31 x 31 x 32
        # ( ( 128 - 16) / 4 ) + 1 = 29. Final is: 29 x 29 x 32
        with tf.variable_scope("cnn1_" + name):
            W_conv1, b_conv1 = self.makeLayerVariables([16, 16, 4, 32], trainable, "conv1")

            h_conv1 = tf.nn.relu(tf.nn.conv2d(x_normalized, W_conv1, strides=[1, 4, 4, 1], padding='VALID') + b_conv1,
                                 name="h_conv1")
            print(h_conv1)

        # Third layer convolves 64 4x4 filters with stride 2 with relu
        # ( ( 31 - 4 ) / 2 ) + 1 = 14. Final is: 14 x 14 x 64
        # ( ( 29 - 7 ) / 2 ) + 1 = 10. Final is: 10 x 10 x 64
        with tf.variable_scope("cnn2_" + name):
            W_conv2, b_conv2 = self.makeLayerVariables([7, 7, 32, 64], trainable, "conv2")

            h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 2, 2, 1], padding='VALID') + b_conv2,
                                 name="h_conv2")
            print(h_conv2)

        # Fourth layer convolves 64 3x3 filters with stride 1 with relu
        # ( ( 12 - 3 ) / 1 ) + 1 = 12. Final is: 12 x 12 x 64
        # ( ( 12 - 3 ) / 1 ) + 1 = 10. Final is: 10 x 10 x 64
        with tf.variable_scope("cnn3_" + name):
            W_conv3, b_conv3 = self.makeLayerVariables([3, 3, 64, 64], trainable, "conv3")

            h_conv3 = tf.nn.relu(tf.nn.conv2d(h_conv2, W_conv3, strides=[1, 1, 1, 1], padding='VALID') + b_conv3,
                                 name="h_conv3")
            print(h_conv3)

        h_conv3_flat = tf.reshape(h_conv3, [-1, 10 * 10 * 64], name="h_conv3_flat")
        print(h_conv3_flat)

        # Fifth layer is fully connected with 512 relu units
        with tf.variable_scope("fc1_" + name):
            W_fc1, b_fc1 = self.makeLayerVariables([10 * 10 * 64, 512], trainable, "fc1")

            h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1, name="h_fc1")
            print(h_fc1)

        # Sixth (Output) layer is fully connected linear layer
        with tf.variable_scope("fc2_" + name):
            W_fc2, b_fc2 = self.makeLayerVariables([512, numActions], trainable, "fc2")

            y = tf.matmul(h_fc1, W_fc2) + b_fc2
            print(y)

        return x, y

    def makeLayerVariables(self, shape, trainable, name_suffix):
        if self.normalizeWeights:
            # This is my best guess at what DeepMind does via torch's Linear.lua and SpatialConvolution.lua (see reset methods).
            # np.prod(shape[0:-1]) is attempting to get the total inputs to each node
            stdv = 1.0 / math.sqrt(np.prod(shape[0:-1]))
            weights = tf.Variable(tf.random_uniform(shape, minval=-stdv, maxval=stdv), trainable=trainable,
                                  name='W_' + name_suffix)
            biases = tf.Variable(tf.random_uniform([shape[-1]], minval=-stdv, maxval=stdv), trainable=trainable,
                                 name='W_' + name_suffix)
        else:
            weights = tf.Variable(tf.truncated_normal(shape, stddev=0.01), trainable=trainable, name='W_' + name_suffix)
            biases = tf.Variable(tf.fill([shape[-1]], 0.1), trainable=trainable, name='W_' + name_suffix)
        return weights, biases

    def inference(self, screens):
        y = self.sess.run([self.y], {self.x: screens})
        q_values = np.squeeze(y)
        return np.argmax(q_values)

    def train(self, s_t, s_t1, rewards, actions, terminals, stepNumber):

        # s_t1
        #x2 = [b.state2.getScreens() for b in batch]
        x2 = s_t1
        y2 = self.y_target.eval(feed_dict={self.x_target: x2}, session=self.sess)

        # s_t
        # x = [b.state1.getScreens() for b in batch]
        x = s_t
        a = np.zeros((len(s_t), self.numActions))
        # y_ is the y_j (expected output)
        y_ = np.zeros(len(s_t))


        for i in range(0, len(s_t)):

            a[i, actions[i]] = 1
            if terminals[i]:
                y_[i] = rewards[i]
            else:
                y_[i] = rewards[i] + gamma * np.max(y2[i])

        # print("--------------PRINT TRAINING:-----------------")
        # print("---A:",a)
        # print("---y_:", y_)
        # print("---y_ shape", y_.shape)
        # print("---y2:", y2)
        # print("---y2 shape", y2.shape)

        curr_train_step, cur_loss, cur_y_a, cur_y = self.sess.run([self.train_step, self.loss, self.y_a, self.y], feed_dict={
            self.x: x,
            self.a: a,
            self.y_: y_
        })

        # print("---y_a:", cur_y_a)
        # print("---y_a shape:", cur_y_a.shape)
        #
        # print("---y:", cur_y)
        # print("---y shape:", cur_y.shape)
        #
        # for i in range(0, len(s_t)):
        #     print("y("+str(i)+"):",cur_y[i,actions[i]] )

        if stepNumber % self.targetModelUpdateFrequency == 0:
            print("---------Q-NETWORK UPDATED!")
            self.sess.run(self.update_target)

        if stepNumber % self.targetModelUpdateFrequency == 0 or stepNumber % self.saveModelFrequency == 0:
            dir = self.baseDir + '/models'
            if not os.path.isdir(dir):
                os.makedirs(dir)
            savedPath = self.saver.save(self.sess, dir + '/model', global_step=stepNumber)

        return cur_loss
