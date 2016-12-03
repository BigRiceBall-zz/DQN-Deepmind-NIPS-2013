import time
import cv2
import numpy                as np
import ale_python_interface as ALE

################################################################################
## The GameEnv class provides a way for the agent to interact with the game
################################################################################
class GameEnv:
    
    ## The GameEnv constructor
    #
    #   @param rom     : The path to the rom to load
    #   @param outSize : An array of two elements [h, w] where h is the height
    #                    of the image to output with the method getScreen and w
    #                    its width
    def __init__(self, rom, outSize):
        self._ale = ALE.ALEInterface()
        self._ale.setInt  ("random_seed".encode(), int(time.time()))
        self._ale.setFloat("repeat_action_probability".encode(), 0)
        self._ale.setBool ("color_averaging".encode(), True)
        self._ale.loadROM (rom.encode())
        
        d = self._ale.getScreenDims()

        ## The size of the screen in the form [height, width]
        self.screenSize = [d[1], d[0]]
        ## The size of the images returned by getScreen  the form
        #  [height, width]
        self.outSize    = outSize
        self._RAWScreen = np.empty([d[0] * d[1]]  , dtype = np.uint8)
        self._RAWScaled = np.empty(self.outSize   , dtype = np.uint8)
        self._RGBScreen = np.empty([d[1], d[0], 3], dtype = np.uint8)
    
    ## The minActions method retuns an array containing the minimal action set
    #  for the loaded game
    #
    #   @return  An array containing the minimum set of legal actions
    def minActions(self):
        return self._ale.getMinimalActionSet()
    
    ## The legActions method returns an array containing the set of legal
    #  actions
    #
    #   @return An array containing the set of legal actions
    def legActions(self):
        return self._ale.getLegalActionSet()
    
    ## The getScreen method converts the game screen to a grayscale array
    #  shrink it to the configured size and returns it
    #
    #   @return A numpy array of numpy.float32 between 0 and 255 and of shape
    #           GameEnv.outSize
    def getScreen(self):
        self._ale.getScreen(self._RAWScreen)

        cv2.resize(src   = self._RAWScreen.reshape(self.screenSize),
                   dst   = self._RAWScaled,
                   dsize = (self.outSize[1], self.outSize[0]))

        return self._RAWScaled.astype(dtype = np.float32, copy = True)

    ## The getScreenRGB method returns an array containing the RBG screen
    #
    #   @return A numpy array of numpy.uint8 of shape
    #           [GameEnv.screenSize[0], GameEnv.screenSize[1], 3]
    def getScreenRGB(self):
        self._ale.getScreenRGB(self._RGBScreen)
        return self._RGBScreen

    ## The act method performs the given action and returns the reward gained
    #  from that action
    #
    #   @return The reward issued from the action taken
    def act(self, action):
        return self._ale.act(action)

    ## The resetGame method reset the game
    def resetGame(self):
        self._ale.reset_game()

    ## The gameOver methods return whether or not the game ended
    #
    #   @return True if the game is in an terminal state, False otherwise
    def gameOver(self):
        return self._ale.game_over()
