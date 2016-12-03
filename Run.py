import sys

import agent.DeepMindAgent as DM 
import Message             as M
import GameEnv             as GE
import Saver               as S
import QtDisplay           as Qt

if len(sys.argv) != 2 :
    print("Usage: {} <path to rom>".format(sys.argv[0]))
    quit()
    

rom    = sys.argv[1]
dbPath = "./data.db"
info   = "The agent is running. Type 'stop' to quit"

m  = M.Message()
s  = S.Saver(dbPath)
e  = GE.GameEnv(rom, [84, 84])
qt = Qt.DisplayHandler()

qt.start()
qt.Qt().createPlotter()
qt.Qt().createEnvDisplay(e, 30)

plt = qt.Qt().plotter()

ag = DM.DeepMindAgent.createNewAgent(m, s, plt, e, "Deepmind NIPS agent")
ag.start()
print(info)
m.write(M.Message.TRAIN, None)

while True:
    cmd = input()
    
    if cmd == "stop":
        break
    else:
        print(info)
        
ag.stopProcessing()
m.write(M.Message.QUIT, None)
print("Waiting for the agent to stop ... ", end = "", flush = True)
ag.join()
print("done")
qt.exit()
