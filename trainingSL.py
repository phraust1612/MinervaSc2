#!/usr/bin/python
# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""train network via supervised learning with replay files"""

import tensorflow as tf
import numpy as np
import dqn

import os
import platform
import sys
import time

from pysc2 import maps
from pysc2 import run_configs
from pysc2.lib import actions as actlib
from pysc2.lib import stopwatch
from pysc2.lib import features
from trainingRL import coordinateToInt

import gflags as flags
from s2clientprotocol import sc2api_pb2 as sc_pb

output_size = 584
screen_size = 64
minimap_size = 64
learning_rate = 0.001

REPLAY_PATH = os.path.expanduser("~") + "/StarCraftII/Replays/"
FLAGS = flags.FLAGS

def train(mainDQN, obs, action, action_spec):
    states = [[obs]]
    if len(action) > 0:
        actions_id = action[0].function
        actions_arg = np.zeros([13],dtype=np.int32)

        arg_index = 0
        for arg in action_spec.functions[actions_id].args:
            if arg.id in range(3):
                actions_arg[arg.id] = coordinateToInt(action[0].arguments[arg_index])
            else:
                actions_arg[arg.id] = (int) (action[0].arguments[arg_index][0])
            arg_index += 1

    else:
        # in case of doing nothing
        actions_id = 0
        actions_arg = np.zeros([13],dtype=np.int32)

    X = states

    Q_target = np.array([actions_id])
    spatial_Q_target = actions_arg

    # y shape : [1, output_size]
    y = mainDQN.predict(states)
    y[np.arange(len(X)), actions_id] = Q_target

    # ySpatial shape : [13, 1, arg_size(id)]
    ySpatial = mainDQN.predictSpatial(states)
    for j in range(13):
        if actions_arg[j] >= 0:
            ySpatial[j][0,actions_arg[j]] = spatial_Q_target[j]

    # Train our network using target and predicted Q values on each episode
    return mainDQN.update(X, y, ySpatial)

def run_loop(replay, player_id, mainDQN):
    """Run SC2 to play a game or a replay."""
    stopwatch.sw.enabled = False
    stopwatch.sw.trace = False

    if not replay:
        sys.exit("Must supply a replay.")

    if replay and not replay.lower().endswith("sc2replay"):
        sys.exit("Replay must end in .SC2Replay.")

    run_config = run_configs.get()

    interface = sc_pb.InterfaceOptions()
    interface.raw = False
    interface.score = True
    interface.feature_layer.width = 24
    interface.feature_layer.resolution.x = screen_size
    interface.feature_layer.resolution.y = screen_size
    interface.feature_layer.minimap_resolution.x = minimap_size
    interface.feature_layer.minimap_resolution.y = minimap_size

    max_episode_steps = 0

    replay_data = run_config.replay_data(replay)
    start_replay = sc_pb.RequestStartReplay(
        replay_data=replay_data,
        options=interface,
        disable_fog=False,
        observed_player_id=player_id)

    with run_config.start(full_screen=False) as controller:
        info = controller.replay_info(replay_data)
        print(" Replay info ".center(60, "-"))
        print(info)
        print("-" * 60)

        map_path = info.local_map_path
        if map_path:
            start_replay.map_data = run_config.map_data(map_path)
        controller.start_replay(start_replay)

        game_info = controller.game_info()
        _features = features.Features(game_info)
        action_spec = _features.action_spec()

        try:
            while True:
                frame_start_time = time.time()
                controller.step(1)
                obs = controller.observe()
                actions = obs.actions
                real_obs = _features.transform_obs(obs.observation)
                real_actions = []
                for action in actions:
                    try:
                        real_actions.append(_features.reverse_action(action))
                    except ValueError:
                        real_actions.append(actlib.FunctionCall(function=0,arguments=[]))
                train(mainDQN, real_obs, real_actions, action_spec)

                if obs.player_result:
                    break
                #time.sleep(max(0, frame_start_time + 1 / FLAGS.fps - time.time()))

        except KeyboardInterrupt:
            pass

        print("Score: ", obs.observation.score.score)
        print("Result: ", obs.player_result)

def _main(unused_argv):
    init = tf.global_variables_initializer()
    replay_list = os.listdir(REPLAY_PATH)

    with tf.Session() as sess:
        mainDQN = dqn.DQN(sess, screen_size, minimap_size, output_size, learning_rate, name="main")
        sess.run(init)

        for replay in replay_list:
            if replay[-10:] != '.SC2Replay':
                continue
            start_time = time.time()
            run_loop(replay, 1, mainDQN)
            run_loop(replay, 2, mainDQN)
            mainDQN.saveWeight()
            print("networks were updated / replay :",replay)
            elapsed_time = time.time() - start_time
            print("Took %.3f seconds... " % (elapsed_time))

if __name__ == "__main__":
    argv = FLAGS(sys.argv)
    sys.exit(_main(argv))
