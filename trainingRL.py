import tensorflow as tf
import numpy as np
from pysc2.env import sc2_env
from collections import deque
from typing import List
import random
import minerva_agent
import dqn
import sys
import time
import gflags as flags
FLAGS = flags.FLAGS

output_size = 584 # no of possible actions
screen_size = 64 # no of possible pixel coordinates
minimap_size = 64

learning_rate = 0.1
discount = 0.99
num_episodes = 100
batch_size = 64
max_buffer_size = 50000
update_frequency = 16

visualize = False
myrace = "P"
botrace = "T"

stored_buffer = deque(maxlen=max_buffer_size)

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

    actions_arg = []
    for x in train_batch:
        action_id = x[1]
        arg_index = 0
        appended = 0
        for arg in env.action_spec().functions[action_id].args:
            if arg.id == 0 or arg.id==1:
                actions_arg.append(minerva_agent.coordinateToInt(x[2][arg_index]))
                appended = 1
                break
            arg_index += 1
        if not appended:
            actions_arg.append(-1)

    X = states

    Q_target = rewards + discount * np.max(targetDQN.predict(next_states), axis=1) * ~done
    spatial_Q_target = rewards + discount * np.max(targetDQN.predictSpatial(next_states), axis=1) *~done

    # y shape : [batch_size, output_size]
    y = mainDQN.predict(states)
    y[np.arange(len(X)), actions_id] = Q_target

    ySpatial = mainDQN.predictSpatial(states)
    for i in range(len(X)):
        if actions_arg[i] >= 0:
            ySpatial[i, actions_arg[i]] = spatial_Q_target[i]

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

            if len(stored_buffer) > batch_size:
                minibatch = random.sample(stored_buffer, batch_size)
                loss, _ = batch_train(env, mainDQN, targetDQN, minibatch)

            if step_count % update_frequency == 0:
                sess.run(copy_ops)
                print("saved...")
                mainDQN.saveWeight()

            state = next_state
            step_count += 1

    except KeyboardInterrupt:
        return timesteps
    finally:
        elapsed_time = time.time() - start_time
        print("Took %.3f seconds for %s steps: %.3f fps" % (
            elapsed_time, total_frames, total_frames / elapsed_time))
    return timesteps

def _main(unused_argv):
    #saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        mainDQN = dqn.DQN(sess, screen_size, minimap_size, output_size, name="main")
        targetDQN = dqn.DQN(sess, screen_size, minimap_size, output_size, name="target")
        sess.run(init)

        copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")
        sess.run(copy_ops)

        for episode in range(num_episodes):
            e = 1.0 / ((episode / 50) + 10) # decaying exploration rate
            with sc2_env.SC2Env(
                    "Odyssey",
                    agent_race=myrace,
                    bot_race=botrace,
                    difficulty="1",
                    visualize=visualize) as env:
                agent = minerva_agent.MinervaAgent(mainDQN)
                run_result = run_loop([agent], env, sess, e, mainDQN, targetDQN, copy_ops, 5000)
                reward = run_result[0].reward
                if reward > 0:
                    env.save_replay("victory/")

            print("%d'th game result :"%(episode+1),reward)
            #saver.save(sess, "sessionSave")

argv = FLAGS(sys.argv)
sys.exit(_main(argv))
