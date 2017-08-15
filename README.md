# MinervaSc2

A machine learning project using DeepMind's [PySC2]("https://github.com/deepmind/pysc2") and [Tensorflow]("https://github.com/tensorflow/tensorflow").

I refered to [Sunghun Kim's repository]("https://github.com/hunkim/ReinforcementZeroToAll/") for DQN class, etc.


## Composition

* trainingRL.py : run main learning loops via DQN.
* minerva.py : contains an agent class which decides actions for every step.
* dqn.py : DQN network class, in order to devide target and learning networks.
