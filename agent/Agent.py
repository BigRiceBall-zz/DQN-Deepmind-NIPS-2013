import threading

import Message         as M
import dqn.ConvNet     as Net
import dqn.Optimizers  as Opt

################################################################################
## The agent class is the base for any intelligent agent
################################################################################
class Agent(threading.Thread):
    ## Agent constructor
    #
    #   @param message : A message object that allows the agent to communicate
    #                    with the main thread
    def __init__(self, message):
        threading.Thread.__init__(self)
        self._m          = message
        self._processing = False


    ## The run method overrides the threading.Thread.run method
    def run(self):
        while True:
            self._m.wait()
            msg = self._m.read()

            if msg == M.Message.QUIT :
                break

            if msg == M.Message.TRAIN :
                self._processing = True
                self._train()

    ## The continueProcessing return wheter the agent should continue 
    #  current job or if it should stop
    #
    #   @return True if the agent should contine, False if it should stop
    def continueProcessing(self):
        return self._processing

    ## The stopProcessing method tells the agent to stop what it's doing
    def stopProcessing(self):
        self._processing = False

    ## The train method is an abstract method. An actual agent should override
    #  this method with one that trains the agent and loop until stopProcessing
    #  is called by the main thread
    def train(self):
        pass

