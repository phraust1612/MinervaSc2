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
from pysc2.lib import app

import gflags as flags
from s2clientprotocol import sc2api_pb2 as sc_pb

REPLAY_HOME = os.path.expanduser("~") + "/StarCraftII/Replays/"
FLAGS = flags.FLAGS

output_size = len(actlib.FUNCTIONS)
flags.DEFINE_string("replay",None,"replay path relative to REPLAY_HOME")
flags.DEFINE_integer("repeat",1,"number of iteration")
flags.DEFINE_bool("win_only", True, "learn only for the player who won if this flag is True")
flags.DEFINE_integer("screen_size", 64, "screen width pixels")
flags.DEFINE_integer("minimap_size", 64, "minimap width pixels")
flags.DEFINE_integer("learning_rate", 0.001, "learning rate")
flags.DEFINE_string("agent_race", "T", "agent race")
flags.DEFINE_string("map_name","AscensiontoAiur", "map name")

def coordinateToInt(coor, size=64):
    return coor[0] + size*coor[1]

def raceToCode(race):
    if race == "R":
        return sc_pb.Random
    elif race == "P":
        return sc_pb.Protoss
    elif race == "T":
        return sc_pb.Terran
    else:
        return sc_pb.Zerg

def mapNameMatch(name:str):
    name = name.replace(' ','')
    name = name.replace('LE','')
    name = name.replace('TE','')
    name = name.lower()
    name2 = FLAGS.map_name.lower()
    return name == name2

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
    interface.feature_layer.resolution.x = FLAGS.screen_size
    interface.feature_layer.resolution.y = FLAGS.screen_size
    interface.feature_layer.minimap_resolution.x = FLAGS.minimap_size
    interface.feature_layer.minimap_resolution.y = FLAGS.minimap_size

    max_episode_steps = 0

    replay_data = run_config.replay_data(replay)
    start_replay = sc_pb.RequestStartReplay(
        replay_data=replay_data,
        options=interface,
        disable_fog=False,
        observed_player_id=player_id)

    with run_config.start(full_screen=False) as controller:
        info = controller.replay_info(replay_data)
        infomap = info.map_name
        inforace = info.player_info[player_id-1].player_info.race_actual
        inforesult = info.player_info[player_id-1].player_result.result
        if FLAGS.map_name and not mapNameMatch(infomap):
            print("map doesn't match, continue...")
            print("map_name:",FLAGS.map_name,"infomap:",infomap)
            return
        if FLAGS.agent_race and raceToCode(FLAGS.agent_race) != inforace:
            print("agent race doesn't match, continue...")
            print("agent_race:",raceToCode(FLAGS.agent_race),"inforace:",inforace)
            return
        if FLAGS.win_only and not inforesult:
            print("this player was defeated, continue...")
            print("result:",inforesult)
            return
        else:
            print("condition's satisfied, training starts :",replay)
            print("map :",infomap)
            print("player id :", player_id)
            print("race :", inforace)
            print("result :", inforesult)

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

def main(unused_argv):
    replay_list = []
    if FLAGS.replay:
        REPLAY_PATH = REPLAY_HOME + FLAGS.replay
    else:
        REPLAY_PATH = REPLAY_HOME

    for root, dirs, files in os.walk(REPLAY_PATH):
        for subdir in dirs:
            tmp = os.path.join(root, subdir)
            if tmp[-10:] == '.SC2Replay':
                replay_list.append(tmp)
        for file1 in files:
            tmp = os.path.join(root, file1)
            if tmp[-10:] == '.SC2Replay':
                replay_list.append(tmp)

    with tf.Session() as sess:
        mainDQN = dqn.DQN(sess, FLAGS.screen_size, FLAGS.minimap_size, output_size, FLAGS.learning_rate, name="main")

        for iter in range(FLAGS.repeat):
            for replay in replay_list:
                start_time = time.time()
                run_loop(replay, 1, mainDQN)
                run_loop(replay, 2, mainDQN)
                mainDQN.saveWeight()
                print("networks were updated / replay :",replay)
                elapsed_time = time.time() - start_time
                print("Took %.3f seconds... " % (elapsed_time))

def _main():
    argv = FLAGS(sys.argv)
    app.really_start(main)

if __name__ == "__main__":
    sys.exit(_main())
