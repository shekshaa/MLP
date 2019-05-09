import tensorflow as tf


class DenseLayer(object):
    def __init__(self, input_dim, output_dim, act,
                 weight_initializer, bias_initializer, stddev=None):
        super(DenseLayer, self).__init__()
        # save input output dimensions
        self.input_dim = input_dim
        self.output_dim = output_dim

        # saving activation function
        self.act = act

        # save initializer
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

        # weights and biases dictionary
        self.vars = {}

        # standard deviation
        self.stddev = stddev

    def __call__(self, inputs):
        x = inputs

        # bias initialization
        self.vars['bias'] = self.bias_initializer(shape=[self.output_dim, ])

        # weight initialization
        if self.stddev is not None:
            self.vars['weight'] = self.weight_initializer(shape=[self.input_dim, self.output_dim],
                                                          stddev=self.stddev)
        else:
            self.vars['weight'] = self.weight_initializer(shape=[self.input_dim, self.output_dim])

        ######################################################
        # TODO: Apply Transformation with weights and biases #
        #   1. Use tf.matmul to multiply inputs by weights   #
        #   2. Add bias                                      #
        #   3. Apply activation function                     #
        #   4. Save final result in transformed variable     #
        ######################################################

        transformed = self.act(tf.matmul(x, self.vars['weight']) + self.vars['bias'])

        ######################################################
        #                    END OF YOUR CODE                #
        ######################################################

        return transformed
