import numpy         as np
import theano        as Th
import theano.tensor as T

###############################################################################
## The RMSProp function implements the RMSProp algorithm
#
#   Given:
#       * \f$f(\boldsymbol{\theta})\f$ : the function to minimize
#       * \f$\boldsymbol{\theta}_t = (\theta_{t,0}, ..., \theta_{t,n-1})\f$ :
#         the value of the parameters at the time step \f$t\f$ that we want to
#         update to minimize \f$f\f$
#       * \f$\boldsymbol{\theta}_0\f$ : the initial value of the \f$\theta\f$
#         parameters
#       * \f$\boldsymbol{m}_t = (m_{t,0}, ..., m_{t,n-1})\f$ : An array of size
#         \f$n\f$ such that \f$m_{0,i} = 0\,\forall i \in \{0, ..., n - 1\}\f$
#       * \f$\boldsymbol{u}_t = (u_{t,0}, ..., u_{t,n-1})\f$ : An array of size
#         \f$n\f$ such that \f$u_{0,i} = 0\,\forall i \in \{0, ..., n - 1\}\f$
#       * \f$\lambda\f$ : The learning rate
#       * \f$\mu\f$ : The momentum
#       * \f$\delta\f$ : The decay
#       * \f$\epsilon\f$ : A small number to avoid dividing by zero
#
#   At each iteration, the RMSProp algorithm performs the following updates
#       1. \f$\boldsymbol{g}_t = \boldsymbol{\nabla_\theta}
#                                 f(\boldsymbol{\theta})|_{\boldsymbol{\theta}=
#                                 \boldsymbol{\theta_{t-1}}}\f$
#       2. \f$m_{t,i} = \delta\,m_{t-1,i} + (1-\delta) g_{t,i}^2\quad\quad
#                       \forall i \in \{0, ..., n - 1\}\f$
#       3. \f$u_{t,i} = \mu\,u_{t-1, i} + \lambda \frac{g_{t,i}}
#                       {\sqrt{m_{t,i} + \epsilon}}\quad\quad
#                       \forall i \in \{0, ..., n - 1\}\f$
#       4. \f$\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} + 
#                                     \boldsymbol{u}_t\f$
#   
#
#   @param grads         : A list of the component of the gradient of the
#                          function to minimize derived with respect to the
#                          parameters to modify
#   @param params        : A list of parameters to update
#   @param learning_rate : The learning rate
#   @param momentum      : The momentum
#   @param decay         : The decay
#   @param epsilon       : A small value to avoid dividing by zero
#
#   @return A list of updates operation to pass to a theano function in order
#           to apply RMSProp algorithm 
#
#   @see The <a href="http://blog.sigopt.com/post/141501625253/sigopt-for-ml-tensorflow-convnets-on-a-budget">SigOpt Blog</a> where the algorithm comes from
###############################################################################
def RMSProp(grads, params, learning_rate = 0.1, momentum = 0.5,
            decay = 0.01, epsilon = 1e-8):

    assert hasattr(grads , "__iter__") and hasattr(grads , "__len__"),   \
           "The parameter 'grads' must be a list of partial derivatives"
    assert hasattr(params, "__iter__") and hasattr(params, "__len__"),   \
           "The parameter 'params' must be a list of parameters"
    assert len(grads) == len(params), \
           "'grads' and 'params' must have the same length"

    lr = learning_rate
    m  = momentum
    d  = decay
    e  = epsilon
        
    updades = []
    i       = 0
        
    for g,p in zip(grads, params):
        m_t  = Th.shared(value = np.zeros(shape = p.get_value().shape,
                                          dtype = np.float32),
                         name  = "m_t[{}]".format(i))
        u_t  = Th.shared(value = np.zeros(shape = p.get_value().shape,
                                          dtype = np.float32),
                         name  = "u_t[{}]".format(i))

        m_t1 = d * m_t + (1.0 - d) * (g ** 2)
        u_t1 = m * u_t + lr * (g / T.sqrt(m_t1 + e))

        updades.append((m_t, m_t1))
        updades.append((u_t, u_t1))
        updades.append((p  , p - u_t1))

        i = i + 1

    return updades

###############################################################################
## The clipByNorm function implements a gradient norm clipping
#
#   @param grads : A gradient list as returned by theano.tensor.grad(...)
#   @param ts    : The threshold
#
#   @return A grads_clipped list which the elements are the elements of 'grads'
#           clipped so that \f$||grads\_clipped|| = \min(ts, ||grads||)\f$
###############################################################################
def clipByNorm(grads, ts):
    assert hasattr(grads , "__iter__") and hasattr(grads , "__len__"),   \
           "The parameter 'grads' must be a list of partial derivatives"
       
    clip  = Th.shared(np.array([ts, 0], dtype = np.float32))
    right = Th.shared(np.array([0 , 1], dtype = np.float32))
    n     = Th.shared(np.array(0     , dtype = np.float32))
    
    for g in grads:
        n = n + (g ** 2).sum()
        
    n    = T.sqrt(n)
    clip = (clip + right * n.dimshuffle('x')).min() / n
    
    grads_clipped = []
    
    for g in grads:
        grads_clipped.append(g * clip)

    return grads_clipped
