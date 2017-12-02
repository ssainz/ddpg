import tensorflow as tf


def build_cnn_network(name, trainable):

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
        W_conv1, b_conv1 = makeLayerVariables([16, 16, 4, 32], trainable, "conv1")

        h_conv1 = tf.nn.relu(tf.nn.conv2d(x_normalized, W_conv1, strides=[1, 4, 4, 1], padding='VALID') + b_conv1,
                             name="h_conv1")
        print(h_conv1)

    # Third layer convolves 64 4x4 filters with stride 2 with relu
    # ( ( 31 - 4 ) / 2 ) + 1 = 14. Final is: 14 x 14 x 64
    # ( ( 29 - 7 ) / 2 ) + 1 = 10. Final is: 10 x 10 x 64
    with tf.variable_scope("cnn2_" + name):
        W_conv2, b_conv2 = makeLayerVariables([7, 7, 32, 64], trainable, "conv2")

        h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 2, 2, 1], padding='VALID') + b_conv2,
                             name="h_conv2")
        print(h_conv2)

    # Fourth layer convolves 64 3x3 filters with stride 1 with relu
    # ( ( 12 - 3 ) / 1 ) + 1 = 12. Final is: 12 x 12 x 64
    # ( ( 12 - 3 ) / 1 ) + 1 = 10. Final is: 10 x 10 x 64
    with tf.variable_scope("cnn3_" + name):
        W_conv3, b_conv3 = makeLayerVariables([3, 3, 64, 64], trainable, "conv3")

        h_conv3 = tf.nn.relu(tf.nn.conv2d(h_conv2, W_conv3, strides=[1, 1, 1, 1], padding='VALID') + b_conv3,
                             name="h_conv3")
        print(h_conv3)

    h_conv3_flat = tf.reshape(h_conv3, [-1, 10 * 10 * 64], name="h_conv3_flat")
    print(h_conv3_flat)

    # Fifth layer is fully connected with 512 relu units
    with tf.variable_scope("fc1_" + name):
        W_fc1, b_fc1 = makeLayerVariables([10 * 10 * 64, 512], trainable, "fc1")

        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1, name="h_fc1")
        print(h_fc1)

    # Sixth (Output) layer is fully connected linear layer
    # with tf.variable_scope("fc2_" + name):
    #     W_fc2, b_fc2 = makeLayerVariables([512, numActions], trainable, "fc2")
    #
    #     y = tf.matmul(h_fc1, W_fc2) + b_fc2
    #     print(y)

    return x, h_fc1

def makeLayerVariables(shape, trainable, name_suffix):
    # if self.normalizeWeights:
    #     # This is my best guess at what DeepMind does via torch's Linear.lua and SpatialConvolution.lua (see reset methods).
    #     # np.prod(shape[0:-1]) is attempting to get the total inputs to each node
    #     stdv = 1.0 / math.sqrt(np.prod(shape[0:-1]))
    #     weights = tf.Variable(tf.random_uniform(shape, minval=-stdv, maxval=stdv), trainable=trainable,
    #                           name='W_' + name_suffix)
    #     biases = tf.Variable(tf.random_uniform([shape[-1]], minval=-stdv, maxval=stdv), trainable=trainable,
    #                          name='W_' + name_suffix)
    # else:
    weights = tf.Variable(tf.truncated_normal(shape, stddev=0.01), trainable=trainable, name='W_' + name_suffix)
    biases = tf.Variable(tf.fill([shape[-1]], 0.1), trainable=trainable, name='W_' + name_suffix)
    return weights, biases