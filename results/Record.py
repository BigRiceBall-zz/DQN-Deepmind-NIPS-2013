import os
import sys
import json
import shutil
import time
import datetime

scriptDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(scriptDir, ".."))

import agent.DeepMindAgent as DM 
import Message             as M
import GameEnv             as GE
import Saver               as S
import QtDisplay           as Qt

################################################################################
## CONFIGURATION
################################################################################
framesFolder = "./pictures/test01/xavier"   # Folder where to store the data
rom          = "../../../ROM/breakout.bin"  # Game to load
dbPath       = "../data_xavier.db"          # Database that stores the agents
agentId      = 1                            # Id of the agent to test
refEpoch     = 100                          # Epoch to replay
################################################################################

env   = GE.GameEnv(rom, [84, 84]) # Environment
msg   = M.Message()               # Unused but required to initialize the agent
saver = S.Saver(dbPath)           # The saver that holds datas about the agents
qt    = Qt.DisplayHandler()       # The display

qt.start()
qt.Qt().createPlotter()
qt.Qt().createEnvDisplay(env, 30)

plt   = qt.Qt().plotter() # The plotter (unsused but required by the agent)

print("Loading the agent ... ", end = "", flush = True)
agent = DM.DeepMindAgent.loadAgent(msg, saver, plt, env, agentId, None)
print("done")

# Delete the directories and their content if their exists an recreate
# empty ones
if os.path.exists(framesFolder + os.sep + "max"):
  shutil.rmtree(framesFolder + os.sep + "max")
  
if os.path.exists(framesFolder + os.sep + "min"):
  shutil.rmtree(framesFolder + os.sep + "min")
  
if os.path.exists(framesFolder + os.sep + "tmp"):
  shutil.rmtree(framesFolder + os.sep + "tmp")
  
os.makedirs(framesFolder + os.sep + "max")
os.makedirs(framesFolder + os.sep + "tmp")

# First play (= max score for now)
print("{:>6} : Playing ... ".format("X"), end = "", flush = True)
t1     = time.time()
maxS,_ = agent.replay(refEpoch, framesFolder + os.sep + "max")
t2     = time.time()
delta  = datetime.timedelta(seconds = int(round(t2 - t1)))
print("done [{}] - [{:0>3}]".format(delta, maxS))

# The maximum score is also the minimum. Copy the created files to the 'min'
# folder
minS = maxS
shutil.copytree(framesFolder + os.sep + "max",
                framesFolder + os.sep + "min")

scores = [maxS] # The list of all the scores
avg    = maxS   # Average score
ep     = 1      # Number of episodes played

# Stat loop
for i in range(1, 1000):
  
  # Play
  print("{:>6} : Playing ... ".format(i), end = "", flush = True)
  t1    = time.time()
  s,_   = agent.replay(125, framesFolder + os.sep + "tmp")
  t2    = time.time()
  delta = datetime.timedelta(seconds = int(round(t2 - t1)))
  print("done [{}] - [{:0>3}]".format(delta, s))
  
  avg = (avg * (ep / (ep + 1))) + (s / (ep + 1))
  ep  = ep  + 1
  scores.append(s)
  
  if s > maxS :
    # Delete the old max
    shutil.rmtree(framesFolder + os.sep + "max")
    # Rename the tmp folder to 'max'
    shutil.move  (framesFolder + os.sep + "tmp",
                  framesFolder + os.sep + "max")
    maxS = s
  elif s < minS:
    # Delete the old min
    shutil.rmtree(framesFolder + os.sep + "min")
    # Rename the tmp folder to 'min'
    shutil.move  (framesFolder + os.sep + "tmp",
                  framesFolder + os.sep + "min")
    minS = s
  else:
    # Remove the newly collected frames
    shutil.rmtree(framesFolder + os.sep + "tmp")
  
  os.makedirs(framesFolder + os.sep + "tmp")
    

print("Max. score      : {}".format(maxS))
print("Min. score      : {}".format(minS))
print("Avg. score      : {}".format(avg))
print("Episodes played : {}".format(ep))

# Save the list of the scores in a JSON file
json.dump(scores, open(framesFolder + os.sep + "scores.json", "w"))

qt.exit()
