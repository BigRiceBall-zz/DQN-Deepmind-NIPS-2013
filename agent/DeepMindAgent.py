import gc
import math
import random
import time
import datetime
import collections       as C
import matplotlib.pyplot as plt
import numpy             as np
import theano            as Th
import theano.tensor     as T

import Plotter           as P
import Saver             as S
import agent.Agent       as A
import dqn.ConvNet       as Net
import dqn.Optimizers    as Opt

import matplotlib        as mpl
mpl.rcParams["backend"]     = "qt4agg"
mpl.rcParams["interactive"] = True

################################################################################
## The ImageSet class implement an image container
#
# The ImageSet class expects that the images to store are numpy arrays of
# np.float32 of shape [h, w] where h is the height of the image and w its width.
#
# If the set if full, this class pre-allocate an array of N images before
# storing new images.
#
# The images stored in the set are actually copied
#
################################################################################
class ImagesSet:
    
    ## The ImageSet constructor
    #
    #   @param chunkSize : The size by witch the set well be increased every
    #                      time it's full and images are inserted
    #   @param h         : The images' height
    #   @param w         : The images' width
    def __init__(self, chunkSize, h, w):
        self._cs           = chunkSize
        self._h            = h
        self._w            = w
        self._chunks       = [] # List of arrays that contains the images
        self._freeSlots    = [] # List of arrays indicating the free slots
        self._fsCnt        = 0  # Total number of free slots
        self._freeTemplate = np.arange(chunkSize)
        
    ## The addImages method adds the given images to the set
    #
    #   @param imgList : An iterable structure containing the images to copy
    #                    to the set
    #
    #   @return A list if slots where the images where stored
    def addImages(self, imgList):
        l   = len(imgList)
        k   = 0
        ret = []
        
        while self._fsCnt < l:
            self._addChunk()
            
        for i,fsl in enumerate(self._freeSlots):
            while len(fsl) > 0:
                j                    = fsl.popleft()
                self._chunks[i][j,:] = imgList[k]
                self._fsCnt          = self._fsCnt - 1
                k                    = k + 1
                ret.append((i,j))
                if k >= l: return ret
            
    ## The image method returns the image stored at the given slot
    #
    #   @param slot : The identifier of the slot where the image is stored
    #
    #   @return The requested image
    def image(self, slot):
        return self._chunks[slot[0]][slot[1]]
    
    ## The free method free the given slots
    #
    #   @param slots : An iterable structure containing the slots to free
    def free(self, slots):
        self._fsCnt = self._fsCnt + len(slots)
        
        for s in slots:
            self._freeSlots[s[0]].append(s[1])
            
    ## The _addChunk method allocate memory to the ImageSet
    def _addChunk(self):
        self._fsCnt  = self._fsCnt + self._cs
        self._chunks.append(np.empty([self._cs, self._h, self._w],
                            dtype = np.float32))
        self._freeSlots.append(C.deque(self._freeTemplate))

###############################################################################
## The ReplayMemory class is a list of past experiences as defined in the paper
#  <a href="https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf"> Playing Atari with
#  Deep Reinforcement Learning</a>
#
#   The replay memory is made of an ImageSet that will keep given images in
#   memory. If 'c' is the number of images a state is made of, the replay memory
#   keeps the ids of the 'c' last inserted images. Then when a new experience
#   is inserted into the replay memory, only one image has to be given. The
#   replay memory assumes that the initial state related to this experience is
#   made of the 'c' last images and that the terminal state of this experience
#   is made of the 'c-1' last inserted images plus the new given image.
#
###############################################################################
class ReplayMemory:
    ## The ReplayMemory constructor initializes a ReplayMemory of the given
    #  capacity
    #
    #   @param capacity : The number of elements the replay memory is able to
    #                     store before erasing its oldest inserted element to
    #                     insert new ones
    #   @param c        : The number of channels (i.e. images) per sequence
    #   @param h        : The heights of an images
    #   @param w        : The width of an image
    #   @param actCnt   : The number of possible actions
    def __init__(self, capacity, actCnt, c, h, w):
        self._actCnt = actCnt
        self._c      = c
        self._h      = h
        self._w      = w
        self._i      = ImagesSet(50000, h, w)
        self._sart   = C.deque(maxlen = capacity)
        self._last   = C.deque(maxlen = c + 1)
       
    ## The __len__ method returns the number of experiences stored in the replay
    #  memory
    def __len__(self):
        return len(self._sart)
        
    ## The addImages method adds the given images to the image set
    #
    #   @param imgs : A list of the images to add to the set
    def addImages(self, imgs):
        slots = self._i.addImages(imgs)
        for s in slots:
            self._last.append(s)
        
    ## Adds the given experience to the replay memory
    #
    #   @param img : The new image to add to the image set
    #   @param a   : The id of the last action taken
    #   @param r   : The last reward perceived
    #   @param t   : Wheter the reached state is a terminal state or not
    def addExperience(self, img, a, r, t):
        slots = self._i.addImages([img])
        self._last.append(slots[0])
        
        if len(self._sart) == self._sart.maxlen:
            oldest = self._sart.popleft()
            
            toFree = None
            if oldest[3] : # if terminal free all images
                toFree = oldest[0]
            else:
                toFree = [oldest[0][0]]
            self._i.free(toFree)
            
        s    = []
        s[:] = self._last
        self._sart.append((s,a,r,t))
        
    ## The minibatch method returns a random minibatch which size is the minimum
    #  between the given size and the size of the replay memory. If the replay
    #  memory is empty, the method returns None
    #
    #   @param size : The size of the minibatch to return
    #
    #   @return This method returns None if the replay memory is empty.
    #           Otherwise it returns a minibatch made of n sample drawn from the
    #           set of stored experiences, where 'n' is the minimum between
    #           the length of the replay memory and the given size.
    #           The returned minibatch is a tuple which elements are the
    #           following:
    #               - s_t  : an array of shape [n, c, h, w] which contains the n
    #                        initial states of the minibatch
    #               - s_t1 : an array of the shape [n, c, h, w] which contains
    #                        the n states reached from respectively the n states
    #                        stored in s_t
    #               - a_t  : an array of shape [n, actCnt] where every row is
    #                        full of 0 except for the value at the id
    #                        corresponding to the action take in the respective
    #                        n states s_t, which is then 1
    #               - r_t  : an array of shape [n] which entries are the rewards
    #                        perceived while going from the n respective states
    #                        s_t to the n respective states s_t1
    #               - t    : an array of shape [n] which entries indicate if
    #                        the n respective states s_t1 are terminals or not
    def minibatch(self, size):
        if len(self) <= 0:
            return None
        
        size   = min(len(self), size)
        sample = random.sample(self._sart, size)
        
        s_t  = np.empty([size, self._c, self._h, self._w], dtype = np.float32)
        s_t1 = np.empty([size, self._c, self._h, self._w], dtype = np.float32)
        a_t  = np.zeros([size, self._actCnt]             , dtype = np.float32)
        r_t  = np.empty([size]                           , dtype = np.float32)
        term = np.empty([size]                           , dtype = np.float32)
        
        for i, sart in enumerate(sample):
            imgs           = [self._i.image(s) for s in sart[0]]
            s_t [i,:]      = imgs[:self._c]
            s_t1[i,:]      = imgs[1:]
            a_t[i,sart[1]] = 1
            r_t[i]         = sart[2]
            term[i]        = sart[3]
            
        return s_t, s_t1, a_t, r_t, term

################################################################################
## The DeepMindAgent class implements the agent described by Deepmind in their
#  article of 2013 "Playing Atari with Deep Reinforcement Learning".
#
#   @see <a href"https://arxiv.org/abs/1312.5602">Playing Atari with Deep
#        Reinforcement Learning</a>
################################################################################
class DeepMindAgent(A.Agent):

    ## The createNewAgent static method returns a new agent with the given name
    #
    #   @param message : The message object that the agent uses to communicate
    #                    whith the main thread
    #   @param saver   : The saver object that the agent uses to save itself
    #   @param plotter : The plotter object the agent can use to plot some datas
    #   @param env     : The gameEnvironement the agent will use as input
    #   @param name    : The name of the agent
    #
    #   @return A new agent that's saved in the saver object
    def createNewAgent(message, saver, plotter, env, name):
        a = DeepMindAgent(message, saver, plotter, env, -1)
        a._newAgent(name)
        return a

    ## The loadAgent static method returns an agent build from a previously
    #  saved agent
    #
    #   @param message : The message object that the agent uses to communicate
    #                    whith the main thread
    #   @param saver   : The saver object that the agent uses to save itself
    #   @param plotter : The plotter object the agent can use to plot some datas
    #   @param env     : The gameEnvironement the agent will use as input
    #   @param agentId : The id of the agent to copy
    #   @param loadNet : The id network to load. If negative (default), the last
    #                    network is loaded. If None, no network is loaded
    #
    #   @return A new agent initilized with the datas previously stored in the
    #           saver object
    def loadAgent(message, saver, plotter, env, agentId, loadNet = -1):
        a = DeepMindAgent(message, saver, plotter, env, agentId)
        a.loadParams()
        
        if not (loadNet is None):
          if loadNet < 0: loadNet = None
          a.loadNetwork(loadNet)
          
        return a 

    ## The agent agent constructor. This shouldn't be directly called. Instread,
    #  the static methods createNewAgent or loadAgent should be used
    #
    #   @param message : The message object that the agent uses to communicate
    #                    whith the main thread
    #   @param saver   : The saver object that the agent uses to save itself
    #   @param plotter : The plotter object the agent can use to plot some datas
    #   @param env     : The gameEnvironement the agent will use as input
    #   @param agentId : The id of the agent
    def __init__(self, message, saver, plotter, env, agentId):
        A.Agent.__init__(self, message)

        ## The id of the agent as used in the saver
        self.id         = agentId
        
        self._env       = env
        self._saver     = saver
        self._plotter   = plotter
        self._networkId = -1
        self._replay    = None
        self._testSet   = None
        self._input     = None

        self._params  = {}
        self._network = {}

    ## The _newAgent method initialize the agent with the default parameters
    #
    #   This method save the agent and its network in the saver object
    #
    #   @param name : The name of the agent
    def _newAgent(self, name):
        act = self._env.minActions().tolist()
        
        # Network parameters
        self._params["N"] = {}
        self._params["N"]["inC"]     = 4                    # Input channels
        self._params["N"]["inH"]     = self._env.outSize[0] # Input height
        self._params["N"]["inW"]     = self._env.outSize[1] # Input width
        self._params["N"]["act"]     = act                  # List of actions
        self._params["N"]["actCnt"]  = len(act)             # Number of actions

        # Optimizer parameters
        self._params["O"] = {}
        self._params["O"]["name"]    = "RMSProp" # Optimizer name
        self._params["O"]["eps"]     = 1e-6      # RMSProp epsilon
        self._params["O"]["mom"]     = 0         # RMSProp momentum
        self._params["O"]["dec"]     = 0.99      # RMSProp decay
        self._params["O"]["lr"]      = 0.00025   # The optimiser learning rate

        # Playing parameters
        self._params["P"] = {}
        self._params["P"]["rep"]     = 4        # Number of actions repeated
        self._params["P"]["wait"]    = 30       # Max. number of frame to wait
        self._params["P"]["obs"]     = 5000     # Observation bef. training

        # Learning parameters
        self._params["L"] = {}
        self._params["L"]["repSize"] = 200000   # Size of the replay memory
        self._params["L"]["maxR"]    = 1        # Maximum reward perceived
        self._params["L"]["minR"]    = -1       # Minimum reward perceived
        self._params["L"]["disc"]    = 0.95     # Discount factor
        self._params["L"]["epsMax"]  = 1        # Maximum value for epsilon
        self._params["L"]["epsMin"]  = 0.1      # Minimum value for epsilon
        self._params["L"]["epsTS"]   = 1000000  # Step when eps reaches its min
        self._params["L"]["batch"]   = 32       # Size of the mini batch

        # Testing parameters
        self._params["T"] = {}
        self._params["T"]["epoch"]    = 50000    # Testing frequency
        self._params["T"]["eps"]      = 0.05     # Epsilon to use for the tests
        self._params["T"]["it"]       = 10000    # Number of steps to test
        self._params["T"]["setMin"]   = 5000     # Minimum size of the test set
        self._params["T"]["setMax"]   = 5000     # Maximum size of the test set
        self._params["T"]["setId"]    = -1       # Test set id
        self._params["T"]["setShape"] = [None,   # Shape of the test set
                                         self._params["N"]["inC"],
                                         self._params["N"]["inH"],
                                         self._params["N"]["inW"]]

        # Internal state
        self._params["S"] = {}
        self._params["S"]["it"]    = 0 # The number of iterations trained
        self._params["S"]["game"]  = 0 # The number of games trianed
        self._params["S"]["cost"]  = 0 # The average cost over iterations
        self._params["S"]["score"] = 0 # The average score over games

        # Network
        self._network["IN"] = {}
        self._network["IN"]["x"] = T.tensor4("Input" , dtype = "float32")
        self._network["IN"]["m"] = T.matrix ("Mask"  , dtype = "float32")
        self._network["IN"]["t"] = T.vector ("Target", dtype = "float32")
        self._network["IN"]["s"] = [None, self._params["N"]["inC"],
                                          self._params["N"]["inH"],
                                          self._params["N"]["inW"]]
        self._network["L1"] = {}
        self._network["L1"]["w"],\
        self._network["L1"]["b"],\
        self._network["L1"]["y"],\
        self._network["L1"]["s"] = Net.ConvLayer(self._network["IN"]["x"],
                                                 self._network["IN"]["s"],
                                                 16, 8, 8, 4, 4,
                                                 Net.Paddings.full,
                                                 Net.ActivationFunctions.relu,
                                                 "L1")
        self._network["L2"] = {}
        self._network["L2"]["w"],\
        self._network["L2"]["b"],\
        self._network["L2"]["y"],\
        self._network["L2"]["s"] = Net.ConvLayer(self._network["L1"]["y"], 
                                                 self._network["L1"]["s"],
                                                 32, 4, 4, 2, 2,
                                                 Net.Paddings.full,
                                                 Net.ActivationFunctions.relu,
                                                 "L2")
        self._network["L3"] = {}
        self._network["L3"]["w"],\
        self._network["L3"]["b"],\
        self._network["L3"]["y"],\
        self._network["L3"]["s"] = Net.FCLayer  (self._network["L2"]["y"],
                                                 self._network["L2"]["s"],
                                                 256,
                                                 Net.ActivationFunctions.relu,
                                                 "L3")
        self._network["L4"] = {}
        self._network["L4"]["w"],\
        self._network["L4"]["b"],\
        self._network["L4"]["y"],\
        self._network["L4"]["s"] = Net.FCLayer  (self._network["L3"]["y"],
                                                 self._network["L3"]["s"],
                                                 self._params["N"]["actCnt"],
                                                 Net.ActivationFunctions.NONE,
                                                 "L4")

        x    = self._network["IN"]["x"]
        m    = self._network["IN"]["m"]
        t    = self._network["IN"]["t"]
        y    = self._network["L4"]["y"]
        p    = [self._network["L1"]["w"], self._network["L1"]["b"],
                self._network["L2"]["w"], self._network["L2"]["b"],
                self._network["L3"]["w"], self._network["L3"]["b"],
                self._network["L4"]["w"], self._network["L4"]["b"]]
        cost = ((t - (y * m).sum(axis = 1)) ** 2).mean()
        grad = Opt.clipByNorm(Th.grad(cost = cost, wrt = p), 1)
        rms  = Opt.RMSProp(grads         = grad, 
                           params        = p, 
                           learning_rate = self._params["O"]["lr"],
                           momentum      = self._params["O"]["mom"],
                           decay         = self._params["O"]["dec"],
                           epsilon       = self._params["O"]["eps"])
        
        self._network["OUT"] = {}
        self._network["OUT"]["y"]     = Th.function(
                                         inputs  = [x],
                                         outputs = [y,
                                                    self._network["L1"]["y"],
                                                    self._network["L2"]["y"],
                                                    self._network["L3"]["y"]])
        self._network["OUT"]["max"]   = Th.function(
                                           inputs  = [x],
                                           outputs = [y.max(axis = 1),
                                                      y.argmax(axis = 1)])
        self._network["OUT"]["avg"]   = Th.function(inputs  = [x],
                                                    outputs = [y.mean(),
                                                               y.max(axis = 1)\
                                                                .mean()])
        self._network["OUT"]["cost"]  = Th.function(
                                           inputs  = [x, m, t],
                                           outputs = cost,
                                           updates = rms)
        self._network["OUT"]["grad"]  = grad
        self._network["OUT"]["rms"]   = rms

        self.id = self._saver.newAgent(name, S.Saver.DEEP_MIND_AGENT, 
                                       self._params)
        self._saveNetwork()
    
    ## The _saveAgent updates the saver with the current agent's parameters
    def _saveAgent(self):
        self._saver.saveAgent(self.id, self._params)
    
    ## The _saveNetwork save the current networks state in the saver object
    def _saveNetwork(self):
        self._networkId = self._saver.saveNetwork(self.id,
                                                  self._params["S"]["it"], 
                                                  self._network)

    ## The loadParams object load the parameters of the current agent saved
    #  in the saver object
    def loadParams(self):
        self._params = self._saver.loadAgent(self.id)
    
    ## The loadNetwork object load the network associated with the given id
    #
    #   @param networkId : The id of the network to load
    def loadNetwork(self, networkId):
        self._networkId = networkId
        self._network   = self._saver.loadNetwork(self.id, networkId)
    
    ## The _train method train the agent and periodically test it
    #
    #   This method start by initializing the replay memory and the test set and
    #   then train until it's asked to stop
    def _train(self):
        if self._replay is None :
            self._initializeReplay()
        
        if self._testSet is None :
            self._initializeTest()

        act        = self._params["N"]["act"]
        actCnt     = self._params["N"]["actCnt"]
        start_time = time.time()
        
        ## Loop until it's asked to stop
        while super().continueProcessing():
            self._newGame()
            self._replay.addImages(self._input)
            score = 0
            ## Loop until the current game ends
            while super().continueProcessing() and \
                  not self._env.gameOver():
                      
                s_t  = np.array(self._input, dtype = np.float32) # Input state
                a_id = self._getNextAction(s_t, self._epsilon()) # Chosen action
                r_t  = self._performAction(act[a_id])            # Reward
                self._updateInput()
                s_t1 = np.array(self._input, dtype = np.float32) # Reached state

                # Clip the reward
                score = score + r_t
                if r_t > 0 : r_t = self._params["L"]["maxR"] 
                if r_t < 0 : r_t = self._params["L"]["minR"]

                # The the current experience to the replay memory
                self._replay.addExperience(self._input[-1], a_id, r_t,
                                           self._env.gameOver())
                
                # Get a new training batch from the memory
                s_j ,\
                s_j1,\
                a_mj,\
                r_j ,\
                t_j  = self._replay.minibatch(self._params["L"]["batch"])
                q_j1 = self._network["OUT"]["max"](s_j1)[0]
                y_j  = r_j + (1 - t_j) * self._params["L"]["disc"] * q_j1

                it   = self._params["S"]["it"]
                it_1 = it + 1

                # Compute the cost for the given minibatch and train the
                # network
                cost_t = float(self._network["OUT"]["cost"](s_j, a_mj, y_j))
                self._params["S"]["cost"] = \
                     self._params["S"]["cost"] * (it / it_1) + (cost_t / it_1)
               
                # Display a line of information in the terminal
                if it % 100 == 0:
                    delta = datetime.timedelta(\
                                seconds = int(round(time.time() - start_time)))
                    print(("{0} - {1:06d} - Sc: {2:5.1f} - e: {3:>6.4f} - " +
                           "Lm: {4:>6.4f} - L: {5:>6.4f} - {6:07}") \
                          .format(delta, it, self._params["S"]["score"],
                                  self._epsilon(),
                                  self._params["S"]["cost"], cost_t,
                                  self._params["S"]["game"]))
                    gc.collect()
                    
                # Test the agent
                if it % self._params["T"]["epoch"] == 0:
                    self._saveAgent()
                    self._saveNetwork()
                    self._test()
                    # Increment the number of iterations and start a new game
                    self._params["S"]["it"] = it_1
                    break

                self._params["S"]["it"] = it_1
                
            g   = self._params["S"]["game"]
            g_1 = g + 1
            self._params["S"]["score"] = \
                        self._params["S"]["score"] * (g / g_1) + (score / g_1)
            self._params["S"]["game"]  = g_1
            
    ## The replay method makes the agent to replay the given epoch
    #
    #  @param epoch : The id of the epoch to replay as it has been recored by
    #                 the saver
    #  @param save  : The path of the folder where to save frames or None to
    #                 disable the recording. Default None
    #  @return A tuple which the first element is the total score and the second
    #          one the total reward
    def replay(self, epoch, save = None):
      
      self.loadNetwork(self._saver.loadNetworkEpoch(self.id, epoch))
      
      score  = 0
      reward = 0
      act    = self._params["N"]["act"]
      
      self._newGame(0)
      while not self._env.gameOver() :
        a_id   = self._getNextAction(self._input)
        r_t    = self._performAction(act[a_id])
        
        score = score + r_t
                
        if r_t > 0 : r_t = self._params["L"]["maxR"]
        if r_t < 0 : r_t = self._params["L"]["minR"]
        
        reward = reward + r_t
        
        if not (save is None):
          self._env.saveFrame(save)
        
        self._updateInput()
      
      return (score, reward)
    
    ## The _test method test the current network
    #
    #   TODO : Instead of testing over a fixed number of iteration, the agent
    #          should be tested over a fixed number of games. Indeed, the more
    #          the agent learn, the more iterations are taken by a game and so
    #          the less games are played. This leads to a situation where we
    #          start by computing statisics over a large number of games and we
    #          end up comparing these early results with the later ones computed
    #          over a small number of games.
    def _test(self):
        print("Testing ... ", end = "", flush = True)
        epoch = self._params["S"]["it"] / self._params["T"]["epoch"]
        
        ## If it's the first epoch, initialize the plotter object
        if epoch == 0:
            style = [{"color":"#5b5bbb", "width" : 2}]
            self._plotter.addGroup("Test")
            self._plotter.addPlot ("Test", "Q Average"     , 1, style)
            self._plotter.addPlot ("Test", "Q Max Average" , 1, style)
            self._plotter.addPlot ("Test", "Average score" , 1, style)
            self._plotter.addPlot ("Test", "Average reward", 1, style)
        
        i     = 0
        chunk = 1000
        avg   = [0, 0]
        while i < len(self._testSet):
            tmp = self._network["OUT"]["avg"](self._testSet[i:i + chunk])
            avg[0] = avg[0] + tmp[0] * len(self._testSet[i:i + chunk])
            avg[1] = avg[1] + tmp[1] * len(self._testSet[i:i + chunk])
            i = i + chunk
            
        avg[0] = avg[0] / len(self._testSet)
        avg[1] = avg[1] / len(self._testSet)
        
        act    = self._params["N"]["act"]
        score  = 0.0
        reward = 0.0
        games  = 0
        it     = 0
        t1     = time.time()
        
        while it < self._params["T"]["it"]:
            self._newGame(0)
            while it < self._params["T"]["it"] and \
                  not self._env.gameOver() :
                a_id   = self._getNextAction(self._input)
                r_t    = self._performAction(act[a_id])
                score  = score + r_t
                
                if r_t > 0 : r_t = self._params["L"]["maxR"]
                if r_t < 0 : r_t = self._params["L"]["minR"]
                reward = reward + r_t
                self._updateInput()
                it = it + 1
            games = games + 1
            
        t2     = time.time()
        delta  = datetime.timedelta(seconds = int(round(t2 - t1)))
        score  = score  / games
        reward = reward / games
        qAvg   = float(avg[0])
        mAvg   = float(avg[1])
        
        print("done [{}]".format(delta))
                                   
        self._plotter.updatePlot("Test", "Q Average"     , epoch, [qAvg]  )
        self._plotter.updatePlot("Test", "Q Max Average" , epoch, [mAvg]  )
        self._plotter.updatePlot("Test", "Average score" , epoch, [score] )
        self._plotter.updatePlot("Test", "Average reward", epoch, [reward])
        
        self._saver.saveStat(self.id, self._networkId, "Q Average"     , epoch,
                             qAvg)
        self._saver.saveStat(self.id, self._networkId, "Q Max Average" , epoch,
                             mAvg)
        self._saver.saveStat(self.id, self._networkId, "Average score" , epoch,
                             score)
        self._saver.saveStat(self.id, self._networkId, "Average reward", epoch,
                             reward)

    ## The _initializeReplay initializes the replay memory and fill it with
    #  random game experiences
    def _initializeReplay(self):
        print("Initializing replay memory ... ", end = "", flush = True)
        self._replay = ReplayMemory(self._params["L"]["repSize"],
                                    self._params["N"]["actCnt"],
                                    self._params["N"]["inC"],
                                    self._params["N"]["inH"],
                                    self._params["N"]["inW"])
        
        act    = self._params["N"]["act"]
        actCnt = self._params["N"]["actCnt"]

        while len(self._replay) < self._params["P"]["obs"] :
            self._newGame()
            self._replay.addImages(self._input)
            while (not self._env.gameOver()) and \
                  (len(self._replay) < self._params["P"]["obs"]) :
                s_t        = np.array(self._input, dtype = np.float32)
                a_id       = random.randrange(actCnt)
                r_t        = self._performAction(act[a_id])
                self._updateInput()
                s_t1       = np.array(self._input, dtype = np.float32)

                if r_t > 0 : r_t = self._params["L"]["maxR"]
                if r_t < 0 : r_t = self._params["L"]["minR"]

                self._replay.addExperience(self._input[-1], a_id, r_t,
                                           self._env.gameOver())
        print("done")
    
    ## The _initializeTest method initilize query the saver for a valid test
    #  set or initializes a new one with random states picked from the game
    #  environment if no test set is currently available
    def _initializeTest(self):
        print("Initializing test set ... ", end = "", flush = True)
        if self._params["T"]["setId"] > 0:
            self._testSet = self._saver.loadDataset(self._params["T"]["setId"])
            print("done")
            return
        
        sl = self._saver.listDatasets(self._params["T"]["setShape"],
                                      self._params["T"]["setMin"],
                                      self._params["T"]["setMax"])
        
        if (not (sl is None)) and (len(sl) > 0):
           self._params["T"]["setId"] = sl[0][0]
           self._testSet              = self._saver.loadDataset(sl[0][0])
           print("done")
           return
        
        self._testSet = np.empty([self._params["T"]["setMin"],
                                  self._params["N"]["inC"],
                                  self._params["N"]["inH"],
                                  self._params["N"]["inW"]],
                                  dtype = np.float32)
        i      = 0
        actCnt = self._params["N"]["actCnt"]
        while i < self._params["T"]["setMin"] :
            self._newGame()
            while (not self._env.gameOver()) and \
                  (i < self._params["T"]["setMin"]):
                
                if random.random() < 0.25 :
                    self._testSet[i,:] = self._input
                    i = i + 1
                self._performAction(random.randrange(actCnt))
                self._updateInput()

        self._params["T"]["setId"] = self._saver.newDataset(
                                                 self._params["T"]["setMin"],
                                                 self._params["T"]["setShape"],
                                                 self._testSet)
        print("done")

    ## The _newGame method resets the game and perform a random number of random
    #  action. At the end, the input state is left in a non terminal state
    #
    #   @param wait : The number of random action to perform after reseting the
    #                 game. If None (default), this number is set to the agent
    #                 waiting parameter
    def _newGame(self, wait = None):
        if self._input is None:
            self._input = C.deque(maxlen = self._params["N"]["inC"])
        
        for i in range(self._input.maxlen):
            self._input.append(np.zeros([self._params["N"]["inH"],
                                         self._params["N"]["inW"]],
                                         dtype = np.float32))
        
        if wait is None:
            wait = self._params["P"]["wait"]
        
        self._env.resetGame()
        actCnt = self._params["N"]["actCnt"]
        w      = random.randint(0, wait)
        i      = 0
        for j in range(w - self._params["N"]["inC"]):
            self._performAction(random.randrange(actCnt))
            i = i + 1
            
            if self._env.gameOver():
                print(("WARNING: Game over after {} actions. " + 
                       "Retry with 'wait' set to {}").format(i, wait - 5))
                return self._newGame(wait - 5)
        
        self._updateInput()
         
        while i < w:
            self._performAction(random.randrange(actCnt))
            self._updateInput()
            i = i + 1
            
            if self._env.gameOver():
                print(("WARNING: Game over after {} actions. " + 
                       "Retry with 'wait' set to {}").format(i, wait - 5))
                return self._newGame(wait - 5)
    
    ## The _updateInput method query the game environment to get the current
    #  screen scale it and push it into the input variable
    def _updateInput(self):
        self._input.append(self._env.getScreen() / 255.0)

    ## The _getNextAction method returns the id of the next action to perform
    #  following an epsilon greedy strategy
    #
    #   @param x       : The initial state
    #   @param epsilon : The value to use for epsilon. If None (default), the
    #                    test value is used
    #
    #   @return The id of the next action to perform
    def _getNextAction(self, x, epsilon = None):
        if epsilon is None :
            epsilon = self._params["T"]["eps"]

        if random.random() < epsilon:
            a_id = random.randrange(self._params["N"]["actCnt"])
        else:
            a_id = self._network["OUT"]["max"]([x])[1][0]

        return a_id

    ## The _performAction method perform the given action 'n' times against
    #  the game environment
    #
    #   @param a : The action to perform
    #
    #   @return The reward accumulated while performinf the actions
    def _performAction(self, a):
        r = 0
        for i in range(self._params["P"]["rep"]):
            r = r + self._env.act(a)
            if self._env.gameOver(): break
        return r

    ## The _epsilon method return the value of epsilon related to the current
    #  iteration
    def _epsilon(self):
        e = (( (self._params["L"]["epsMin"] - self._params["L"]["epsMax"]) \
              / self._params["L"]["epsTS"]) * self._params["S"]["it"])     \
            + self._params["L"]["epsMax"]

        return max(self._params["L"]["epsMin"], e)

    ## The _displayImage method displays the given images
    #
    #   @param imgs : An array of images to display with matplotlib.pyplot
    def _displayImage(self, imgs):
        w         = math.ceil(math.sqrt(len(imgs)))
        fig, axes = plt.subplots(w, w)
        fig.subplots_adjust(hspace = 1/w, wspace = 1/w)

        for i, ax in enumerate(axes.flat):
            if(i < len(imgs)):
                ax.imshow(imgs[i])
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis("off")

        plt.show(block=True)
