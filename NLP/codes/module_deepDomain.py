import logging
import time
import math
import numpy as np
import theano
import theano.tensor as T
import state_parser_deepDomain as sp

from model_deepDomain import Model
from utils_deepDomain import *
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool
import codecs

logger = logging.getLogger(__name__)

def add_to_params(params, new_param):
    params.append(new_param)
    return new_param

class Module(object):
    def __init__(self, state, rng):
        self.state = state
        self.rng = rng
        
        self.params=[]
    
def softrelu(x):
    return T.switch(x<0, 0.01*x, x)

def relu(x):
    return T.switch(x<0, 0, x)
    
class CharPhraseQuantization(Module):
    """ character embedding """
    def __init__(self, state, rng, srng, n_in, n_out, c_val, np_in):
        logger.debug('Initializing Character Quantization Layer: {0} x {1}'.format(n_in + np_in, n_out+(np_in*2)))
        
        Module.__init__(self, state, rng)
        self.srng = srng
        self.init_params(n_in, n_out, c_val, np_in)
        
    def init_params(self, n_in, n_out, c_val, np_in):
        C_values = np.asarray(
                self.rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
        for c in c_val:
            for j in range(n_out):
                C_values[c_val[c][0]][j] = c_val[c][2][j]

        C_emb = theano.shared(value=C_values, name='C_emb', borrow=True)
        
        P_values = np.asarray(
                self.rng.uniform(
                    low=-np.sqrt(6. / (np_in + np_in)),
                    high=np.sqrt(6. / (np_in + np_in)),
                    size=(np_in, np_in * 2)
                ),
                dtype=theano.config.floatX
            )

        P_emb = theano.shared(value=P_values, name='P_emb', borrow=True)

        self.c_emb =add_to_params(self.params, C_emb)
        self.p_emb =add_to_params(self.params, P_emb)
    
    def build(self, samples, phrase):   
        c_emb_output = self.c_emb[samples]
        p_emb_output = T.dot(phrase, self.p_emb)
        #p_emb_output = self.p_emb[phrase]
        emb_output = T.concatenate([c_emb_output, p_emb_output], axis=3)
      
        return emb_output  

class DropOutCharPhraseQuantization(Module):
    """ character embedding """
    def __init__(self, state, rng, srng, n_in, n_out, c_val, np_in, dropout = 0):
        logger.debug('Initializing Drop Out Character Quantization Layer: {0} x {1}'.format(n_in + np_in, n_out+(np_in*2)))
        
        Module.__init__(self, state, rng)
        self.srng = srng
        self.dropout = dropout
        self.init_params(n_in, n_out, c_val, np_in)
        
    def init_params(self, n_in, n_out, c_val, np_in):
        C_values = np.asarray(
                self.rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
        for c in c_val:
            for j in range(n_out):
                C_values[c_val[c][0]][j] = c_val[c][2][j]

        C_emb = theano.shared(value=C_values, name='C_emb', borrow=True)
        
        P_values = np.asarray(
                self.rng.uniform(
                    low=-np.sqrt(6. / (np_in + np_in)),
                    high=np.sqrt(6. / (np_in + np_in)),
                    size=(np_in, np_in * 2)
                ),
                dtype=theano.config.floatX
            )

        P_emb = theano.shared(value=P_values, name='P_emb', borrow=True)

        self.c_emb =add_to_params(self.params, C_emb)
        self.p_emb =add_to_params(self.params, P_emb)
    
    def build(self, samples, phrase, is_train=1):   
        c_emb_output = self.c_emb[samples]
        #p_emb_output = self.p_emb[phrase]
        p_emb_output = T.dot(phrase, self.p_emb)
        emb_output = T.concatenate([c_emb_output, p_emb_output], axis=3)

        output = T.switch(T.eq(is_train, 1), self._dropout_from_layer(emb_output), emb_output)

        return output

    def _dropout_from_layer(self, layer):
        """p is the probablity of dropping a unit
        """
        # p=1-p because 1's indicate keep and p is prob of dropping
        mask = self.srng.binomial(n=1, p=self.dropout, size=layer.shape, dtype=theano.config.floatX)
        # The cast is important because
        # int * float32 = float64 which pulls things off the gpu
        output = (layer*(1./self.dropout)) * mask

        return output
    
class StackInfoEmb(Module):
    """ character embedding """
    def __init__(self, prefix, state, rng, srng, n_in, n_out):
        logger.debug('Initializing Stack Embedding Layer: {0} x {1}'.format(n_in, n_out))
        Module.__init__(self, state, rng)
        self.srng = srng
        self.init_params(prefix, n_in, n_out)

    def init_params(self, prefix, n_in, n_out):
        S_values = np.asarray(
                self.rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )

        S_emb = theano.shared(value=S_values, name='S_emb_'+prefix, borrow=True)

        self.s_emb =add_to_params(self.params, S_emb)


    def build(self, samples,is_train=1):
        lin_output = self.s_emb[samples]
        self.output = T.switch(T.eq(is_train,1), lin_output, lin_output*0.75)
 #       self.output = lin_output

        return self.output
    
class LeNetConvPoolLayer(Module):
    """Pool Layer of a convolutional network """

    def __init__(self, nth, state, rng, filter_shape, image_shape, poolsize, border_mode="valid") :
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type samples: theano.tensor.dtensor4
        :param samples: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num samples feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num samples feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """        
        logger.debug('Initializing LeNetConvPoolLayer {0}: {1} -> {2} x {3} / {4} '.format(nth, image_shape[2], filter_shape[0], filter_shape[2], poolsize[0]))
        Module.__init__(self, state, rng)
        self.init_params(nth, filter_shape, image_shape, poolsize)
        
    def init_params(self, nth, filter_shape, image_shape, poolsize, border_mode="valid"):        

        assert image_shape[1] == filter_shape[1]

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) // np.prod(poolsize))
        # initialize weights with random weights
        wname = "conv_" + nth + "_w"
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = add_to_params(self.params, theano.shared(
                    np.asarray(
                            self.rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                            dtype=theano.config.floatX
                            ),
                    name=wname,
                    borrow=True
                )
            )
             
        # the bias is a 1D tensor -- one bias per output feature map
        bname = "conv_" + nth + "_b"
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = add_to_params(self.params, theano.shared(value=b_values, name=bname, borrow=True) )
        
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize

                 
    def build(self, samples):   

        # convolve input feature maps with filters
        conv_out = conv2d(
            input=samples,
            filters=self.W,
            filter_shape=self.filter_shape,
            image_shape=self.image_shape
        )
        # downsample each feature map individually, using maxpooling
        #pooled_out = downsample.max_pool_2d(
        pooled_out = pool.pool_2d(
            input=conv_out,
            ds=self.poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        #return T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        #return T.nnet.relu(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        return softrelu(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
    
    def convolution(self, samples):

        # convolve input feature maps with filters
        conv_out = conv2d(
            input=samples,
            filters=self.W,
            filter_shape=self.filter_shape,
            image_shape=self.image_shape
        )
        #return T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        return softrelu(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
              
    def pooling(self, samples):
        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=samples,
            ds=self.poolsize,
            ignore_border=True
        )
        
        return pooled_out        

class FullyConnectedLayer(Module):
    def __init__(self, nth, state, rng, srng, n_in, n_out, W=None, b=None, activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """        
        logger.debug('Initializing Fully Connected Layer{0}: {1} x {2}'.format(nth, n_in, n_out))
        Module.__init__(self, state, rng)    
        self.srng = srng
        self.init_params(nth,  n_in, n_out, W=None, b=None, activation=T.tanh)
        
    def init_params(self, nth, n_in, n_out, W=None, b=None, activation=T.tanh):      
        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        wname = "hd_" + str(nth) + "_w"
        if W is None:
            W_values = np.asarray(
                self.rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name=wname, borrow=True)
            self.W =add_to_params(self.params, W)
    
        bname = "hd_" + str(nth) + "_b"
        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name=bname, borrow=True)
            self.b = add_to_params(self.params, b)

        self.activation = activation

    
    def build(self, samples): 

        lin_output = T.dot(samples, self.W) + self.b
            
        layer_output = ( 
                       lin_output if self.activation is None
                       else self.activation(lin_output)
                       )
            
        return layer_output 
    
class DropOutFullyConnectedLayer(Module):
    def __init__(self, nth, state, rng, srng, n_in, n_out, dropout = 0, W=None, b=None, activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """        
        if dropout> 0 : logger.debug('Initializing Drop Out Fully Connected Layer {0}: {1} x {2}'.format(nth, n_in, n_out))
        else: logger.debug('Initializing Fully Connected Layer{0}: {1} x {2}'.format(nth, n_in, n_out))
        Module.__init__(self, state, rng)    
        self.srng = srng
        self.dropout = dropout
        self.init_params(nth,  n_in, n_out, W=None, b=None, activation=T.tanh)
        
    def init_params(self, nth, n_in, n_out, W=None, b=None, activation=T.tanh):      
        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        wname = "hd_" + str(nth) + "_w"
        if W is None:
            W_values = np.asarray(
                self.rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name=wname, borrow=True)
            self.W =add_to_params(self.params, W)
    
        bname = "hd_" + str(nth) + "_b"
        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name=bname, borrow=True)
            self.b = add_to_params(self.params, b)

        self.activation = activation

    def build(self, samples, is_train=1):

        lin_output = T.dot(samples, self.W) + self.b

        layer_output = (
                       lin_output if self.activation is None
                       else self.activation(lin_output)
                       )

        output = T.switch(T.eq(is_train, 1), self._dropout_from_layer(layer_output), layer_output)

        return output

    def _dropout_from_layer(self, layer):
        """p is the probablity of dropping a unit
        """
        # p=1-p because 1's indicate keep and p is prob of dropping
        mask = self.srng.binomial(n=1, p=self.dropout, size=layer.shape, dtype=theano.config.floatX)
        # The cast is important because
        # int * float32 = float64 which pulls things off the gpu
        output = (layer*(1./self.dropout)) * mask

        return output

class LogisticRegression(Module):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """
    def __init__(self, prefix, state, rng, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        logger.debug('Initializing {0} Logistic regression Layer'.format(prefix))
        Module.__init__(self, state, rng)        
        self.init_params(prefix, n_in, n_out)   

    def init_params(self, prefix, n_in, n_out):        
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        W = theano.shared(value=numpy.zeros( (n_in, n_out), dtype=theano.config.floatX ),
                          name='dec_W_'+prefix,
                          borrow=True
                          )
        # initialize the biases b as a vector of n_out 0s
        b = theano.shared(value=numpy.zeros((n_out,), dtype=theano.config.floatX),
                          name='dec_b_'+prefix,
                          borrow=True
                          )
        self.W =add_to_params(self.params, W)
        self.b = add_to_params(self.params, b)
        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        
    def build(self, samples):

        p_y_given_x = T.nnet.softmax(T.dot(samples, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        y_pred = T.argmax(p_y_given_x, axis=1)
        
        return p_y_given_x, y_pred

    def buildWithMask(self, samples, mask):
        
        #befor_mask = T.nnet.softmax(T.dot(samples, self.W) + self.b)
        #self.p_y_given_x = befor_mask*mask
        
        p_y_given_x_only = T.nnet.softmax(T.dot(samples, self.W) + self.b + mask)
        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        y_pred_only = T.argmax(p_y_given_x_only, axis=1)
        
        return p_y_given_x_only, y_pred_only
    
    def negative_log_likelihood(self, p_y_given_x, y, w):
        """Return the mean of the negative log-likelihood of the prediction
        of this model_deepintent under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        e = 1e-7
        p_y_given_x = T.clip(p_y_given_x, e, 1-e)
        return -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]), y] * w)
    
    def negative_log_likelihood_with_penalty(self, p_y_given_x, y, label, penalty):

        revised = T.set_subtensor(p_y_given_x[:,label], p_y_given_x[:,label]*penalty)
        e = 1e-7
        revised = T.clip(revised, e, 1-e)
        return -T.mean(T.log(revised)[T.arange(y.shape[0]), y])
    
    def errors(self, y_pred, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """
        # check if y has same dimension of y_pred
        if y.ndim != y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(y_pred, y))
        else:
            raise NotImplementedError()
                   
class BatchNormalization(object) :
    def __init__(self, layername, input_shape, mode=1 , momentum=0.9) :
        '''
        # params :
        input_shape :
            when mode is 0, we assume 2D input. (mini_batch_size, # features)
            when mode is 1, we assume 4D input. (mini_batch_size, # of channel, # row, # column)
        mode : 
            0 : feature-wise mode (normal BN)
            1 : window-wise mode (CNN mode BN)
        momentum : momentum for exponential average
        '''
        logger.debug('Initializing Batch Normalization Layer: {0}'.format(input_shape[1]))
        self.input_shape = input_shape
        self.mode = mode
        self.momentum = momentum

        self.insize = input_shape[1]
        
        # random setting of gamma and beta, setting initial mean and std
        rng = np.random.RandomState(int(time.time()))
        self.gamma = theano.shared(np.asarray(rng.uniform(low=-1.0/math.sqrt(self.insize), high=1.0/math.sqrt(self.insize), size=(input_shape[1])), dtype=theano.config.floatX), name=layername+'_gamma', borrow=True)
        self.beta = theano.shared(np.zeros((input_shape[1]), dtype=theano.config.floatX), name=layername+'_beta', borrow=True)
        self.mean = theano.shared(np.zeros((input_shape[1]), dtype=theano.config.floatX), name=layername+'_mean_nograd', borrow=True)
        self.var = theano.shared(np.ones((input_shape[1]), dtype=theano.config.floatX), name=layername+'_var_nograd', borrow=True)

        # parameter save for update
        self.params = [self.gamma, self.beta, self.mean, self.var]

    def build(self, input_value, is_train=1) :
        # returns BN result for given input_value.
        epsilon = 1e-06
        self.mean_up = self.mean
        self.var_up = self.var
        
        if self.mode==0 :
            output = T.switch(T.eq(is_train, 1), self._m0_update_mean_var(input_value, epsilon), self._m0_no_update_mean_var(input_value, epsilon))

        else : 
            output = T.switch(T.eq(is_train, 1), self._m1_update_mean_var(input_value, epsilon), self._m1_no_update_mean_var(input_value, epsilon))
            
        return output, self.mean_up, self.var_up
    
    def _m0_update_mean_var(self, input_value, epsilon):
        now_mean = T.mean(input_value, axis=0)
        now_var = T.var(input_value, axis=0)
        now_normalize = (input_value - now_mean) / T.sqrt(now_var+epsilon) # should be broadcastable..
        output = self.gamma * now_normalize + self.beta
        # mean, var update
        self.mean_up = self.momentum * self.mean + (1.0-self.momentum) * now_mean
        if (self.input_shape[0] == 1) :
            self.var_up = self.momentum * self.var + (1.0-self.momentum) * ((self.input_shape[0]+1)/self.input_shape[0]*now_var)
        else:
            self.var_up = self.momentum * self.var + (1.0-self.momentum) * (self.input_shape[0]/(self.input_shape[0]-1)*now_var)  
              
        return output
    
    def _m0_no_update_mean_var(self,input_value, epsilon):
          
        output = self.gamma * (input_value - self.mean) / T.sqrt(self.var+epsilon) + self.beta
        
        return output
    
    def _m1_update_mean_var(self, input_value, epsilon):
        now_mean = T.mean(input_value, axis=(0,2,3))
        now_var = T.var(input_value, axis=(0,2,3))
        # mean, var update
        self.mean_up = self.momentum * self.mean + (1.0-self.momentum) * now_mean
        self.var_up = self.momentum * self.var + (1.0-self.momentum) * (self.input_shape[0]/(self.input_shape[0]-1)*now_var)
              
        # change shape to fit input shape
        now_mean = self.change_shape(now_mean)
        now_var = self.change_shape(now_var)
        now_gamma = self.change_shape(self.gamma)
        now_beta = self.change_shape(self.beta)
            
        output = now_gamma * (input_value - now_mean) / T.sqrt(now_var+epsilon) + now_beta              
        
        return output
    
    def _m1_no_update_mean_var(self,input_value, epsilon):  
        now_mean = self.mean
        now_var = self.var
        
        # change shape to fit input shape
        now_mean = self.change_shape(now_mean)
        now_var = self.change_shape(now_var)
        now_gamma = self.change_shape(self.gamma)
        now_beta = self.change_shape(self.beta)
            
        output = now_gamma * (input_value - now_mean) / T.sqrt(now_var+epsilon) + now_beta     
                
        return output
        
    # changing shape for CNN mode
    def change_shape(self, vec) :
        return T.repeat(vec, self.input_shape[2]*self.input_shape[3]).reshape((self.input_shape[1],self.input_shape[2],self.input_shape[3]))
