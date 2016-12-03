import threading as Thr

################################################################################
## The Message class overrides the threading.Event class to provide a simple
#  way of communication between threads
################################################################################
class Message(Thr.Event):

    ## Default message type sent to aks the listenner to terminate
    QUIT  = 0
    ## Default message type sent to ask the listenner to start trainind
    TRAIN = 1

    ## The Message class constructor
    def __init__(self):
        Thr.Event.__init__(self)
        self._msg  = None
        self._data = None

    ## The read method reads the current message and reset the threading.Event
    #  internal flag
    #
    #   @return The registered message
    def read(self) :
        super().clear()
        return self._msg

    ## The write method writes a message and set the threading.Event internal
    #  flag
    #
    #   @param message : The message to send.
    #   @param data    : Any additional datas associated to the message. This
    #                    is not used for now
    def write(self, message, data):
        self._msg  = message
        self._data = data
        super().set()
