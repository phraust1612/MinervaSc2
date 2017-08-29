import tensorflow as tf
import numpy as np
from pysc2.env import sc2_env
from pysc2.lib import actions as actlib
from pysc2.lib import app
from collections import deque
from typing import List
import random
import minerva_agent
import dqn
import sys
import time
import gflags as flags
import psutil
import resource
FLAGS = flags.FLAGS

output_size = len(actlib.FUNCTIONS) # no of possible actions
flags.DEFINE_integer("start_episode", 0, "starting episode number")
flags.DEFINE_integer("num_episodes", 100, "total episodes number")
flags.DEFINE_integer("screen_size", 64, "screen width pixels")
flags.DEFINE_integer("minimap_size", 64, "minimap width pixels")

flags.DEFINE_integer("learning_rate", 0.001, "learning rate")
flags.DEFINE_integer("discount", 0.99, "discount factor")
flags.DEFINE_integer("batch_size", 16, "size of mini-batch")
flags.DEFINE_integer("max_buffer_size", 50000, "maximum deque size")
flags.DEFINE_integer("update_frequency", 16, "update target frequency")

flags.DEFINE_bool("visualize", False, "visualize")
flags.DEFINE_string("agent_race", "T", "agent race")
flags.DEFINE_string("bot_race", "R", "bot race")
flags.DEFINE_string("map_name","AscensiontoAiur", "map name")
flags.DEFINE_string("difficulty","1", "bot difficulty")

# below is a list of possible map_name
# AbyssalReef
# Acolyte
# AscensiontoAiur
# BelShirVestige
# BloodBoil
# CactusValley
# DefendersLanding
# Frost
# Honorgrounds
# Interloper
# MechDepot
# NewkirkPrecinct
# Odyssey
# PaladinoTerminal
# ProximaTerminal
# Sequencer

def coordinateToInt(coor, size=64):
    return coor[0] + size*coor[1]

def batch_train(env, mainDQN, targetDQN, train_batch: list) -> float:
    """Trains `mainDQN` with target Q values given by `targetDQN`
    Args:
        mainDQN (dqn.DQN): Main DQN that will be trained
        targetDQN (dqn.DQN): Target DQN that will predict Q_target
        train_batch (list): Minibatch of stored buffer
            Each element is (s, a, r, s', done)
            [(state, action, reward, next_state, done), ...]
    Returns:
        float: After updating `mainDQN`, it returns a `loss`
    """
    states = np.vstack([x[0] for x in train_batch])
    actions_id = np.array([x[1] for x in train_batch])
    rewards = np.array([x[3] for x in train_batch])
    next_states = np.vstack([x[4] for x in train_batch])
    done = np.array([x[5] for x in train_batch])

    # actions_arg[i] : arguments whose id=i
    actions_arg = np.ones([13,FLAGS.batch_size],dtype=np.int32)
    actions_arg *= -1

    batch_index = 0
    for x in train_batch:
        action_id = x[1]
        arg_index = 0

        for arg in env.action_spec().functions[action_id].args:
            if arg.id in range(3):
                actions_arg[arg.id][batch_index] = coordinateToInt(x[2][arg_index])
            else:
                actions_arg[arg.id][batch_index] = (int) (x[2][arg_index][0])
            arg_index += 1
        batch_index += 1

    X = states

    Q_target = rewards + FLAGS.discount * np.max(targetDQN.predict(next_states), axis=1) * ~done
    spatial_Q_target = []
    spatial_predict = targetDQN.predictSpatial(next_states)
    for i in range(13):
        spatial_Q_target.append( rewards + FLAGS.discount * np.max(spatial_predict[i], axis=1) *~done )

    # y shape : [batch_size, output_size]
    y = mainDQN.predict(states)
    y[np.arange(len(X)), actions_id] = Q_target

    # ySpatial shape : [13, batch_size, arg_size(id)]
    ySpatial = mainDQN.predictSpatial(states)
    for j in range(13):
        for i in range(len(X)):
            if actions_arg[j][i] >= 0:
                ySpatial[j][i][actions_arg[j][i]] = spatial_Q_target[j][i]

    # Train our network using target and predicted Q values on each episode
    return mainDQN.update(X, y, ySpatial)


def get_copy_var_ops(*, dest_scope_name: str, src_scope_name: str) -> List[tf.Operation]:
    """Creates TF operations that copy weights from `src_scope` to `dest_scope`
    Args:
        dest_scope_name (str): Destination weights (copy to)
        src_scope_name (str): Source weight (copy from)
    Returns:
        List[tf.Operation]: Update operations are created and returned
    """
    # Copy variables src_scope to dest_scope
    op_holder = []

    src_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder

# returns pysc2.env.environment.TimeStep after end of the game
def run_loop(agents, env, sess, e, mainDQN, targetDQN, copy_ops, max_frames=0):
    total_frames = 0
    stored_buffer = deque(maxlen=FLAGS.max_buffer_size)
    start_time = time.time()

    action_spec = env.action_spec()
    observation_spec = env.observation_spec()
    for agent in agents:
        agent.setup(observation_spec, action_spec)

    timesteps = env.reset()
    state = timesteps[0].observation
    step_count = 0

    for a in agents:
        a.reset()
    try:
        while True:
            total_frames += 1
            if np.random.rand(1) < e:
                # choose a random action and explore
                actions = [agent.step(timestep, 0)
                       for agent, timestep in zip(agents, timesteps)]
            else:
                # choose an action by 'exploit'
                actions = [agent.step(timestep, 1)
                       for agent, timestep in zip(agents, timesteps)]

            if max_frames and total_frames >= max_frames:
                return timesteps

            timesteps = env.step(actions)
            next_state = timesteps[0].observation
            reward = timesteps[0].reward
            done = timesteps[0].last()

            if done:
                break

            stored_buffer.append( (state, actions[0].function, actions[0].arguments, reward, next_state, done) )

            if len(stored_buffer) > FLAGS.batch_size:
                minibatch = random.sample(stored_buffer, FLAGS.batch_size)
                loss, _ = batch_train(env, mainDQN, targetDQN, minibatch)

            if step_count % FLAGS.update_frequency == 0:
                sess.run(copy_ops)

            state = next_state
            step_count += 1

    except KeyboardInterrupt:
        return timesteps
    finally:
        elapsed_time = time.time() - start_time
        print("Took %.3f seconds for %s steps: %.3f fps" % (
            elapsed_time, total_frames, total_frames / elapsed_time))
    return timesteps

def main(unusued_argv):
    parent_proc = psutil.Process()
    with tf.Session() as sess:
        mainDQN = dqn.DQN(sess, FLAGS.screen_size, FLAGS.minimap_size, output_size, FLAGS.learning_rate, name="main")
        targetDQN = dqn.DQN(sess, FLAGS.screen_size, FLAGS.minimap_size, output_size, FLAGS.learning_rate, name="target")

        copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")
        sess.run(copy_ops)
        print("memory before starting the iteration : %s (kb)"%(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

        for episode in range(FLAGS.start_episode, FLAGS.num_episodes):
            e = 1.0 / ((episode / 50) + 2.0) # decaying exploration rate
            with sc2_env.SC2Env(
                    FLAGS.map_name,
                    screen_size_px=(FLAGS.screen_size, FLAGS.screen_size),
                    minimap_size_px=(FLAGS.minimap_size, FLAGS.minimap_size),
                    agent_race=FLAGS.agent_race,
                    bot_race=FLAGS.bot_race,
                    difficulty=FLAGS.difficulty,
                    visualize=FLAGS.visualize) as env:

                agent = minerva_agent.MinervaAgent(mainDQN)
                run_result = run_loop([agent], env, sess, e, mainDQN, targetDQN, copy_ops, 5000)
                agent.close()
                reward = run_result[0].reward
                if reward > 0:
                    env.save_replay("victory/")
                #else:
                #    env.save_replay("defeat/")

            children = parent_proc.children(recursive=True)
            for child in children:
                print("remaining child proc :", child)
            print("memory after exit %d'th sc2env : %s (kb)"%(episode, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

            mainDQN.saveWeight()
            print("networks were saved, %d'th game result :"%episode,reward)

def _main():
    argv = FLAGS(sys.argv)
    app.really_start(main)

if __name__ == "__main__":
    sys.exit(_main())
