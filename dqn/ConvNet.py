import numbers
import math
import enum               as E
import numpy              as np
import theano             as Th
import theano.tensor      as T
import theano.tensor.nnet as Net

###############################################################################
## The InitMethod inner is an enumeration of valid initialization for the
#  functions 'weights' and 'biases'
###############################################################################
class InitMethod(E.Enum):
    ## Initialize the variables with the given value
    constant = 0
    ## Perform a xavier initialization on the variables
    xavier   = 1
    ## initialize the variables following a uniform distribution
    uniform  = 2
    ## initialize the variables following a normal distribution
    normal   = 3
    
    ## The isValid method check is the given value is a valid enumerated value
    #
    #   @param e : The value to check
    #
    #   @return True if the given value is a valid initialization method or
    #           False otherwise
    def isValid(e):
        return (e == InitMethod.constant) or \
               (e == InitMethod.xavier)   or \
               (e == InitMethod.uniform) or \
               (e == InitMethod.normal)


###############################################################################
## The ActivationFunctions class is an enumaration of valid activation 
#  funcitons
###############################################################################
class ActivationFunctions(E.Enum):
    ## Use the Identity activation function
    NONE    = None
    ## Use a ReLU activation function
    relu    = Net.relu
    ## Use a softmax activation function
    softmax = Net.softmax

    ## The isValid method check is the given value is a valid enumerated value
    #
    #   @param e : The value to check
    #
    #   @return True if the given value is a valid activation function or
    #           False otherwise
    def isValid(e):
        return (e == ActivationFunctions.NONE)    or \
               (e == ActivationFunctions.relu)    or \
               (e == ActivationFunctions.softmax)


###############################################################################
## The Paddings class enumerates the valid values for padding
#   @see 
#   <a href="http://deeplearning.net/software/theano_versions/dev/library/tensor/nnet/conv.html#theano.tensor.nnet.conv2d">theano.tensor.nnet.conv2d</a>
###############################################################################
class Paddings(E.Enum):
    valid = "valid"
    full  = "full"
    half  = "half"
    
    ## The isValid method check is the given value is a valid enumerated value
    #
    #   @param e : The value to check
    #
    #   @return True if the given value is a valid padding value or False 
    #           otherwise
    def isValid(e):
        return isinstance(e, int)           or \
               (hasattr(e, "__len__") and \
                (len(e) == 2)         and \
                isinstance(e[0], int) and \
                isinstance(e[1], int)     ) or \
               (e == Paddings.valid)        or \
               (e == Paddings.full )        or \
               (e == Paddings.half )
    
    ## The outputShape compute the shape of the output of a convolutional layer
    #  set up with the given attributes
    #
    #   @param inputShape : The shape of the input passed to the layer
    #   @param filters    : The number of filters
    #   @param fH         : The height of the filters
    #   @param fW         : The width of the filters
    #   @param vS         : The vertical stride
    #   @param hS         : The horizontal stride
    #   @param p          : The padding applied to the input before performing
    #                       the convolution
    #
    #   @return A four entries list representing the computed shape
    def outputShape(inputShape, filters, fH, fW, vS, hS, p):
        assert Paddings.isValid(p), \
               "The padding '{}' is invalid".format(p)
        
        vP = 0
        hP = 0
        
        if   p == Paddings.valid : vP = 0      ; hP = 0
        elif p == Paddings.full  : vP = fH - 1 ; hP = fW - 1
        elif p == Paddings.half  : vP = fH // 2; hP = fW // 2
        elif isinstance(p, int)  : vP = e      ; hP = e
        else                     : vP = e[0]   ; hP = e[1]
        
        d = []
        d.append(inputShape[0])
        d.append(filters)
        d.append(((inputShape[2] + 2 * vP - fH) // vS) + 1)
        d.append(((inputShape[3] + 2 * hP - fW) // hS) + 1)
        
        ## DEBUG
        #print("{0} "                                            + \
        #      "[{1} | {2} - {3} | {4} - {5} | {6} : {7} - {8}]" + \
        #      "\t=> {9}"\
        #      .format(inputShape, filters, fH, fW, vS, hS, p, vP, hP, p))
        
        return d


###############################################################################
## The shared function returns a shared variable of the type float32
#   
#   @param shape   : The shape of the variable to create
#   @param method  : The initialization method to use. This must be one of
#                    the values enumerated in InitMethod
#   @param initArg : A value to use to initialize the variable. It's
#                    effects depend on the 'method' param
#   @param name    : The name of the variable
#   <table>
#   <tr>
#       <th>method</th>
#       <th>initArg</th></tr>
#   <tr>
#       <td>InitMethod.constant</td>
#       <td>The constant value to assign to the the elements 
#           of the variable</td>
#   </tr>
#   <tr>
#       <td>InitMethod.xavier</td>
#       <td>The number of inputs</td>
#   </tr>
#   <tr>
#       <td>InitMethod.uniform</td>
#       <td>A tuple containing the lower and the upper boundaries</td>
#   </tr>
#   <tr>
#       <td>InitMethod.normal</td>
#       <td>A tuple containing the mean and the standard deviation</td>
#   </tr>
#   </table>
#
#   @return A Theano shared variable
###############################################################################
def shared(shape, method, initArg, name):
    assert InitMethod.isValid(method), \
           "The given initialization method is not valid"
      
    init = None
    
    if method == InitMethod.constant:
        assert isinstance(initArg, numbers.Number), \
               "The 'constant' initialization method expects a number " + \
               "parameter"
        init = np.full(shape, initArg, dtype = np.float32)
    
    elif method == InitMethod.uniform:
        assert hasattr(initArg, "__len__")            and \
               len(initArg) == 2                      and \
               isinstance(initArg[0], numbers.Number) and \
               isinstance(initArg[1], numbers.Number),    \
               "The 'uniform' initialization method expects a pair of " + \
               "numbers as parameter"
        init = np.asarray(np.random.uniform(initArg[0], initArg[1], shape),
                          dtype = np.float32)
    
    elif method == InitMethod.xavier:
        assert isinstance(initArg, int), \
               "The 'xavier' initialization method expects an interger " + \
               "parameter"
        init = np.asarray(np.random.normal(0, np.sqrt(2 / initArg), shape),
                          dtype = np.float32)
    
    elif method == InitMethod.normal:
        assert hasattr(initArg, "__len__")            and \
               len(initArg) == 2                      and \
               isinstance(initArg[0], numbers.Number) and \
               isinstance(initArg[1], numbers.Number),    \
               "The 'normal' initialization method expects a pair of " + \
               "numbers as parameter"
        init = np.asarray(np.random.normal(initArg[0], initArg[1], shape),
                          dtype = np.float32)
    
    return Th.shared(value = init, name = name)


###############################################################################
## The FCLayer function returns a the weights, the biases and the output of a
#  fully connected layer and the shape of its output
#
#   @param layerIn    : The input of the layer
#   @param inputShape : A list containing the dimensions of the input in
#                       the form <tt>[ batch, d0, d1, ..., dn ]</tt>
#   @param filters    : The number of filters
#   @param act        : The activation function. This must be one of the
#                       value enumerated in Layer.ActivationFunctions
#   @param name       : The name of the layer
#   @param clipMin    : The minimum value allowed for the gradient compoments
#                       associated to this layer. If None (default), the
#                       clipping is disabled
#   @param clipMax    : The maximum value allowed for the gradient compoments
#                       associated to this layer. If None (default), the
#                       clipping is disabled
#
#   @returns A tuple containing the newly created variables, the output and
#            its shape : (w, b, y, shape)
###############################################################################
def FCLayer(layerIn, inputShape, filters, act, name,
            clipMin = None, clipMax = None):
    assert ActivationFunctions.isValid(act), \
           "The given activation function is not valid"
    
    d = 1
    for e in inputShape[1:]:
        d = d * e
    
    w = shared([d, filters], InitMethod.xavier  , d, name + "_W")
    b = shared([filters]   , InitMethod.constant, 0, name + "_B")
    
    if (not (clipMin is None)) and (not (clipMax is None)):
        w_c = Th.gradient.grad_clip(w, clipMin, clipMax)
        b_c = Th.gradient.grad_clip(b, clipMin, clipMax)
    else:
        w_c = w
        b_c = b
    
    z = T.dot(layerIn.flatten(2), w_c) + b_c.dimshuffle('x', 0)
    
    y = None
    if act == ActivationFunctions.NONE :
        y = z
    else:
        y = act(z)
    
    outputShape = [inputShape[0], filters]
    
    return w, b, y, outputShape


###############################################################################
## The ConvLayer function returns a the weights, the biases and the output of a
#  convolutional layer and the shape of its output
#
#   @param layerIn    : The input of the layer
#   @param inputShape : The shape of the input of the layer. This must be
#                       a list with four dimensions :
#                       <tt>[ batch, channels, height, width ]</tt>
#   @param filters    : The number of filters
#   @param fHeight    : The filters' height
#   @param fWidth     : The filters' width
#   @param vStride    : The convolution vertical stride
#   @param hStride    : The convolution horizontal stride
#   @param inPad      : The type of padding to apply. It must be one of
#                       the value enumarated in ConvNet.Paddings, an int
#                       or a pair of int
#   @param act        : The activation function of the layer. Is must be
#                       one of the value enumarated in
#                       'ActivationFunctions'
#   @param name       : The name of the layer
#   @param clipMin    : The minimum value allowed for the gradient compoments
#                       associated to this layer. If None (default), the
#                       clipping is disabled
#   @param clipMax    : The maximum value allowed for the gradient compoments
#                       associated to this layer. If None (default), the
#                       clipping is disabled
#
#   @returns A tuple containing the newly created parameters, the output and
#            its shape : (w, b, y, shape)
###############################################################################
def ConvLayer(layerIn, inputShape, filters, fHeight, fWidth,
              vStride, hStride, inPad, act, name,
              clipMin = None, clipMax = None) :
    assert ActivationFunctions.isValid(act), \
           "The given activation function is not valid"
    
    assert Paddings.isValid(inPad), \
           "The padding '{}' is not valid".format(inPad)
    
    
    wShape = [filters, inputShape[1], fHeight, fWidth]
    i      = wShape[1] * wShape[2] * wShape[3]
    w      = shared(wShape   , InitMethod.xavier  , i, name + "_W")
    b      = shared([filters], InitMethod.constant, 0, name + "_B")
    
    if (not (clipMin is None)) and (not (clipMax is None)):
        w_c = Th.gradient.grad_clip(w, clipMin, clipMax)
        b_c = Th.gradient.grad_clip(b, clipMin, clipMax)
    else:
        w_c = w
        b_c = b
        
    z = Net.conv2d(input        = layerIn,
                   filters      = w_c,
                   input_shape  = inputShape,
                   filter_shape = wShape,
                   border_mode  = inPad.value,
                   subsample    = (vStride, hStride)) + \
        b_c.dimshuffle('x', 0, 'x', 'x')
    
    y = None
    if act == ActivationFunctions.NONE :
        y = z
    else:
        y = act(z)
    
    outputShape = Paddings.outputShape(inputShape, filters, fHeight   , fWidth,
                                       vStride   , hStride, inPad)
    
    return w, b, y, outputShape

