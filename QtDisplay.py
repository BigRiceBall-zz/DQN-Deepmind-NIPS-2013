import time
import datetime
import threading    as Thr
import PyQt4.QtCore as qtc
import PyQt4.QtGui  as qtg
import Plotter      as P
import EnvDisplay   as ED

## The QDisplay class provides a way to integrate different window or Qt
#  applications. This class is a QtCore.QObject that hold the QApplication
#  and starts the event loop when the 'run' method is called
class QDisplay(qtc.QObject):

    ## The create_plotter_sig signal signals the the thread that manages the
    #  display to create a new plotter object
    create_plotter_sig     = qtc.pyqtSignal()
    ## The create_env_display_sig signal signals the the thread that manages the
    #  display to create a new environment display object
    create_env_display_sig = qtc.pyqtSignal(object, int)
    
    ## The QDisplay constructor initializes a new QDisplay object. There should
    #  only be one and only one QDisplay object
    def __init__(self):
        qtc.QObject.__init__(self)
        self._app      = None
        self._plt      = None
        self._env      = None
        self._pltReady = Thr.Event()
        self._envReady = Thr.Event()
        self._pltReady.clear()
        self._envReady.clear()
        
    ## The run method initializes the object and connect the signals to their
    #  handlers. Then, it set the given event and start the event loop
    #
    #   @param ready : A threading.Event object that will be set after the
    #                  signals are connected
    def run(self, ready):
        self.initialize()
        self.connectSignals()
        ready.set()
        self._app.exec_()

    ## The initialize method initializes the application
    def initialize(self):
        self._app = qtg.QApplication([])
    
    ## The connectSignals method connects the signals to their handlers
    def connectSignals(self):
        self.create_plotter_sig    .connect(self._createPlotter)
        self.create_env_display_sig.connect(self._createEnvDisplay)
        
    ## The _createPlotter method initilizes the plotter object manage by this
    #  class
    def _createPlotter(self):
        if self._plt is None:
            self._plt = P.Plotter()
            self._pltReady.set()
            
    ## The _createEnvDisplay method initilizes the EnvDisplay object manage by
    #  this class
    def _createEnvDisplay(self, env, freq):
        if self._env is None:
            self._env = ED.EnvDisplay(env, freq)
            self._envReady.set()
        
    ## The createPlotter method sends a signal to the thread that manage the
    #  display in order to ask him to create a plotter object
    def createPlotter(self):
        self.create_plotter_sig.emit()
        
    ## The createEnvDisplay method sends a signal to the thread that manage the
    #  display in order to ask him to create an EnvDisplay object
    def createEnvDisplay(self, env, freq):
        self.create_env_display_sig.emit(env, freq)
        
    ## The plotter method wait for the plotter object to be initialized and
    #  returns the object
    #
    #   @return A reference to the plotter object managed by this class
    def plotter(self):
        self._pltReady.wait()
        return self._plt
    
    ## The envDisplay method wait for the envDisplay object to be initialized
    #  and returns the object
    #
    #   @return A reference to the envDisplay object managed by this class
    def envDisplay(self):
        self._envReady.wait()
        return self._env
        
    ## The exit method force the event loop to quit
    def exit(self):
        self._app.exit()
        
## The DisplayHandler class implements a thread that will be responsible of
#  managing any windows or other display or Qt events
class DisplayHandler(Thr.Thread):

    ## The DisplayHandler constructor initialize the thread
    def __init__(self):
        Thr.Thread.__init__(self)
        self._Qt      = None
        
        ## An event that indicates all the signals are connected
        self.qt_ready = Thr.Event()
        self.qt_ready.clear()
        
    ## The run method initilizes the Qt display and start the event loop
    def run(self):
        Thr.current_thread().name = "QT_D"
        self._Qt = QDisplay()
        self._Qt.run(self.qt_ready)
        
    ## The Qt methods wait for the Qt display to be reday and returns it
    #
    #   @return A reference to the QDisplay object managed by this thread
    def Qt(self):
        self.qt_ready.wait()
        return self._Qt
        
    ## The exit method force the QDisplay object to leave its event loop and
    #  wait for the current thread to terminate
    def exit(self):
        self._Qt.exit()
        self.join()
