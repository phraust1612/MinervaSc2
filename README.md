# MinervaSc2

A machine learning project using DeepMind's [PySC2](https://github.com/deepmind/pysc2) and [Tensorflow](https://github.com/tensorflow/tensorflow).

I refered to [Sunghun Kim's repository](https://github.com/hunkim/ReinforcementZeroToAll/) for DQN class, etc.


## Usage

At first, please download my git and some prerequisites.

Here's my example.

```shell
git clone https://github.com/phraust1612/MinervaSc2.git
sudo pip3 install pysc2
sudo pip3 install numpy
sudo pip3 install tensorflow
```

Specify your saving directory. For default, DQN structure will be saved at 'saved/',

so be sure that 'saved/' directory exists or prepare for your own.

To run reinforcement learning process,

```shell
python3 trainingRL.py (starting episode number) (total episodes)
```

For default, starting episode and total number would be 0 and 100

To run supervised learning process with your replays,

```shell
python3 trainingSL.py
```

For default option, this will refer ~/StarCraftII/Replays/ for replay files

## Composition

* trainingRL.py : run reinforcement learning loops via DQN.
* trainingSL.py : run supervised learning loops with your replay files.
* minerva_agent.py : contains an agent class which decides actions for every step.
* dqn.py : DQN network class, in order to devide target and learning networks.
