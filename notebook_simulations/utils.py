import numpy as np
import tensorflow as tf
import collections

NNetParam = collections.namedtuple('NNetParam', ['ws', 'bs'])


def offset_zip(indexable, offset=1):
    return zip(indexable[:-offset], indexable[offset:])


def shuffle_in_unison_inplace(a, b):
    assert a.shape[0] == b.shape[0]
    p = np.random.permutation(a.shape[0])
    return a[p], b[p]


def init_nn_params(layer_sizes, std=.1):
    """Initialize layers on the neural net.

    Initialize each layers by:
    1. Create right number of Tensorflow variables.
    2. Initialize these variables with Gaussian noise.

    Args:
        layer_sizes ([int]) : the size of each layer (including intput and output layers).
        std (float) : the standard deviation.

    Returns:
        [NNetParam]
    """
    ws = [tf.Variable(tf.random_normal(size, stddev=std))
          for size in offset_zip(layer_sizes, 1)]
    bs = [tf.Variable(tf.zeros([n_out, ]))
          for _, n_out in offset_zip(layer_sizes, 1)]
    return NNetParam(ws=ws, bs=bs)


def feedforward_net(op_in, params, activate_fns):
    """Construct the neural network from all the parameters.

    ReLu function is used as activation function for each layer.

    Args:
        X (Variable) : the input variable.
        params: the parameters of each layers

    Returns:
        The Node that compute the final output.
    """
    op = op_in
    if activate_fns is None:
        activate_fns = (tf.nn.relu for _ in params.ws)
    
    for w, b, activate_fn in zip(params.ws, params.bs, activate_fns):
        op = tf.matmul(op, w) + b
        if activate_fn is not None:
            op = activate_fn(op)
    return op

# TODO support layer-wise activate_fn setting.

class FeedForwardNet:
    def __init__(self, layer_sizes, activate_fns=None):
        self.params = init_nn_params(layer_sizes)
        self.layer_sizes = layer_sizes
        self.activate_fns = activate_fns
        
    def out(self, in_op):
        return feedforward_net(in_op, self.params, self.activate_fns)

    @property
    def all_parameters(self):
        return self.params.ws + self.params.bs
