# DQN-Deepmind-NIPS-2013
This project is an implementation of the deep Q network described by deepmind in [their article of 2013][DM_PAPER] using python 3.5 and Theano

---

#### Dependencies

---

##### Main dependencies
The following dependencies are crucial for this project and are used by the agent itself. The given version should be used. However, higher version would probaly work too
* __Arcade learning environment (ALE)__: the versions 0.5.0 and 0.5.1 downloadable on the [ALE website][ALE_WEB] don't support python 3. However the current development version (December 2016) available on their [Github repository][ALE_GIT] does.
* __Theano__: The release version 0.8 and the current 0.9 development version (December 2016) seems to differ slightly as some packages/modules seems to have changed names or been moved. I used the 0.9 develpment version available from the [Theano Github repository][THEANO_GIT].
* __cv2__: The cv2 module is used to resize the images from ALE to feed the network used by the agent. The version used is the 3.1.0

##### Other dependencies
These dependencies are used by
* __PyQt__: PyQt version 4.8.7 is used to display the game and to build the window where the plots are displayed.
* __pyqtgraph__: The plots and percentile plots are displayed thanks to pyqtgraph version 0.10.0.
* __sqlite3__: The agent's data and the statistics collected during the training are saved in an sqlite database.
* __matplotlib__: matplotlib is not really used and is there for debug purpose only. This should be removed in the future
* __Usual python modules__: collections, datetime, enum, gc, json, math, numbers, numpy, pickle, random, sys, time, threading


---

#### Usage

---
To start training the agent, simply type `python Run.py <rom file>`
This will create a new default agent, initilize it and start training it. A window will open displaying the game in the form it's feeded to the agent and another windows will show the evolution of the agent accross the epochs.
To stop the process, type `stop` and every processes will terminate.

The Atari 2600 ROMs are available on the [AtariAge website ][ATARI]. After downloading the file you have to make sure its named as ALE expects it to be named otherwise, it can lead a segmentation fault. The names for each supported games can be found in the ALE sources, by inspecting the file related to your game in [/src/games/supported/][ALE_SRC].

---

#### Results

---
The agent is trained as explained in the [deepmind paper][DM_PAPER].

Before training, a test set is built by picking random samples from games played randomly. By default, one epoch lasts 50000 iterations and every epoch, the agent plays for 10000 iterations using an epsilon greedy strategy with epsilon equal to 0.05. Every epoch, the following results are plotted and stored:
* The average value of the output of the network over the test set
* The average value of the maximum outputs of the network over the test set
* The average score per games played
* The average reward per games played (as in the Deepmind's paper, the reward are clipped between -1 and 1)

These results are stored in an sqlite database. [DB Browser for SQLite][DB_BROWSER] provides an easy way to display and plot those results one ce the program stopped.

While I didn't observe the same evolution of the output of the Q function as deepmind, I got similar results for the average scores.

---

#### References

---
[1] [Playing Atari with Deep Reinforcement Learning][DM_PAPER]

[2] [Arcade Learning Environlent Technical Manual][ALE_MAN]

[3] [CS231n: Convolutional Neural Networks for Visual Recognition - course notes][CS231n]

[CS231n]: http://cs231n.github.io/
[DB_BROWSER]: https://github.com/sqlitebrowser/sqlitebrowser
[DM_PAPER]: https://arxiv.org/abs/1312.5602
[ALE_WEB]: http://www.arcadelearningenvironment.org/downloads/
[ALE_GIT]: https://github.com/mgbellemare/Arcade-Learning-Environment
[ALE_SRC]: https://github.com/mgbellemare/Arcade-Learning-Environment/tree/master/src/games/supported
[ALE_MAN]: https://github.com/mgbellemare/Arcade-Learning-Environment/blob/master/doc/manual/manual.pdf
[THEANO_GIT]: https://github.com/Theano/Theano
[ATARI]: http://www.atariage.com/system_items.html?SystemID=2600&ItemTypeID=ROM
