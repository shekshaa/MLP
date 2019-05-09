import tensorflow as tf
from layers import DenseLayer


flags = tf.app.flags
FLAGS = flags.FLAGS


class Dense(object):
    def __init__(self, num_hidden, weight_initializer, bias_initializer,
                 act=tf.nn.sigmoid, logging=False, stddev=None):

        super(Dense, self).__init__()

        # saving batch info in placeholders
        self.placeholders = {'batch_images': tf.placeholder(shape=[None, 32 * 32], dtype=tf.float32),
                             'batch_labels': tf.placeholder(shape=[None, 10], dtype=tf.float32)}

        # storing input dimensions of all layers
        self.num_hidden = [self.placeholders['batch_images'].shape.as_list()[1]] + num_hidden

        # setting types of weight and bias initializer
        self.stddev = stddev
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

        # setting activation functions
        self.act = act

        # build layers
        self.layers = []
        self._build()

        # list of activations
        self.activations = [self.placeholders['batch_images']]
        for layer in self.layers:
            ##########################################################################
            # TODO: Forward Propagation                                              #
            # Apply layer on previous layer output                                   #
            # Note that last layer output is last element of self.activations list   #
            # Append output of current layer to the self.activations list            #
            ##########################################################################

            hidden = layer(self.activations[-1])
            self.activations.append(hidden)

            ##########################################################################
            #                             END OF YOUR CODE                           #
            ##########################################################################

        # output of last activations
        self.output = self.activations[-1]

        # defining loss and accuracy and optimizer
        self.loss = self._loss()
        self.acc = self._accuracy()

        # log vars
        if logging:
            self.log_vars()

        #################################################################################
        # TODO: Define Optimizer                                                        #
        # Use GradientDescent Optimizer                                                 #
        # Use FLAGS.learning_rate defined in jupyter notebook as initial learning rate  #
        # Apply optimizer on self.loss to minimize it and save it on self.training      #
        #################################################################################

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
        self.training = self.optimizer.minimize(self.loss)

        #################################################################################
        #                             END OF YOUR CODE                                  #
        #################################################################################

    def log_vars(self):
        for i, layer in enumerate(self.layers):
            ##########################################################################
            # TODO: Add Histogram Summary for all layers                             #
            # For each layer add its bias and weights in tf.summary                  #
            # Please set name of each tensor (weight or bias) based on layer number  #
            # Example: Like bias_1 (bias of layer 1) or weight_2 (weight of layer 2) #
            ##########################################################################

            tf.summary.histogram(name='bias_{}'.format(i + 1), values=layer.vars['bias'])
            tf.summary.histogram(name='weight_{}'.format(i + 1), values=layer.vars['weight'])

            ##########################################################################
            #                            END OF YOUR CODE                            #
            ##########################################################################

    def _build(self):
        for i in range(1, len(self.num_hidden)):
            # set last layer activation as linear function otherwise use self.act
            if i == len(self.num_hidden) - 1:
                act = (lambda x: x)
            else:
                act = self.act

            #######################################################
            # TODO: Add a DenseLayer Object as a layer            #
            # Use DenseLayer class to define a new layer          #
            # Please set all its constructor arguments properly   #
            # These arguments include:                            #
            #   1. Input and output dimensions                    #
            #   2. Weight and Bias Initializer (stddev if needed) #
            #   3. Activation function                            #
            #######################################################

            layer = DenseLayer(input_dim=self.num_hidden[i - 1],
                               output_dim=self.num_hidden[i],
                               weight_initializer=self.weight_initializer,
                               bias_initializer=self.bias_initializer,
                               stddev=self.stddev,
                               act=act)

            ########################################################
            #                   END OF YOUR CODE                   #
            ########################################################

            # add layer to layers list
            self.layers.append(layer)

    def _loss(self):
        #################################################################################################
        # TODO: Calculating Loss function                                                               #
        # Loss function has two general terms:                                                          #
        #   1. Cross Entropy loss over all instances                                                    #
        #   2. Regularization Loss                                                                      #
        # Use tf.nn.softmax_cross_entropy_with_logits to evaluate loss over all instances               #
        # Apply tf.nn.l2_loss on weights of first two layers to evaluate l2 loss                        #
        # Add all the above losses in variable avg_loss                                                 #
        # Finally add loss as a scalar in tf.summary                                                    #
        #################################################################################################

        # cross-entropy loss over logits and labels
        batch_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.output,
                                                             labels=self.placeholders['batch_labels'])

        # L2 regularization on weights
        l2_loss = tf.nn.l2_loss(self.layers[0].vars['weight']) + tf.nn.l2_loss(self.layers[1].vars['weight'])
        l2_loss = FLAGS.weight_decay * l2_loss

        # compute average of batch loss
        avg_loss = tf.reduce_mean(batch_loss) + l2_loss

        # save summary scalar
        tf.summary.scalar('loss', avg_loss)

        #################################################################################################
        #                                        END OF YOUR CODE                                       #
        #################################################################################################

        return avg_loss

    def _accuracy(self):
        #################################################################################################
        # TODO: Calculating Accuracy                                                                    #
        # Calculate prediction over batch instances using tf.argmax on self.output                      #
        # Calculate correct labels from one-hot true labels stored in self.placeholders['batch_labels'] #
        # Use tf.equal to find instances predicted correctly                                            #
        # Use tf.reduce_mean to evaluate accuracy (Be careful about type casting)                       #
        # Finally add accuracy as a scalar in tf.summary                                                #
        #################################################################################################

        # prediction of output on batch
        batch_predictions = tf.argmax(self.output, axis=1)

        # true labels
        correct_predictions = tf.argmax(self.placeholders['batch_labels'], axis=1)

        # check equality of prediction and true labels
        mask = tf.equal(batch_predictions, correct_predictions)

        # compute accuracy using reduce_mean
        avg_acc = tf.reduce_mean(tf.cast(mask, dtype=tf.float32))

        # save summary scalar
        tf.summary.scalar('acc', avg_acc)

        #################################################################################################
        #                                        END OF YOUR CODE                                       #
        #################################################################################################

        return avg_acc
