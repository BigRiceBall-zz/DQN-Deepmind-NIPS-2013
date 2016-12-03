import numpy        as np
import PyQt4.QtGui  as qtg
import PyQt4.QtCore as qtc

################################################################################
## The Canvas class is a QtGui.QWidget that display an image
################################################################################
class Canvas(qtg.QWidget):
    
    ## The Canvas class constructor
    #
    #   @param env : The GameEnv object to use to retrieve the game screen
    def __init__(self, env):
        qtg.QWidget.__init__(self)
        self._env  = env
        self._img  = None
        self._cmap = [qtg.qRgb(i, i, i) for i in range(0, 256)]

    ## The drawScreen methods order the Canvas object to update the image to
    #  display with the current screen from the game environment
    def drawScreen(self):
        self._img = qtg.QImage(self._env.getScreen().astype(np.uint8),
                               self._env.outSize[1],
                               self._env.outSize[0],
                               qtg.QImage.Format_Indexed8)
        self._img.setColorTable(self._cmap)
        super().update()

    ## The paintEvent method overrides the QtGui.QWidget method and simply
    #  repaint the current image
    #
    #   @param event
    def paintEvent(self, event):
        if not (self._img is None):
            painter = qtg.QPainter(self)
            painter.drawImage(0, 0, self._img)
    
################################################################################
## The EnvDisplay class is responsible of displaying and periodically refresh
#  the display of the game environment
################################################################################
class EnvDisplay(qtc.QObject):

    ## The EnvDisplay constructor
    #
    #   @param env  : the GameEnv object to use
    #   @param freq : refresh rate
    def __init__(self, env, freq = 30):
        qtc.QObject.__init__(self)

        self._env   = env
        self._freq  = freq
        self._win   = Canvas(self._env)

        self._win.resize(self._env.outSize[1], self._env.outSize[0])
        
        self._timer = qtc.QTimer()
        self._timer.setInterval(1000 / self._freq)
        self._timer.timeout.connect(self._win.drawScreen)
        
        self._win.show()
        self._timer.start()
              
        
