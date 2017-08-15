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

input_size = minerva_agent.STATE_SIZE # no of possible states
output_size = 584 # no of possible actions
screen_size = 64*64 # no of possible pixel coordinates
minimap_size = 64*64

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

def batch_train(env, DQNlist, train_batch: list) -> float:
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
    mainDQN = DQNlist[0]
    targetDQN = DQNlist[1]
    mainScreenDQN = DQNlist[2]
    targetScreenDQN = DQNlist[3]
    mainMinimapDQN = DQNlist[4]
    targetMinimapDQN = DQNlist[5]

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
    screenQ_target = rewards + discount * np.max(targetScreenDQN.predict(next_states), axis=1) * ~done
    minimapQ_target = rewards + discount * np.max(targetMinimapDQN.predict(next_states), axis=1) * ~done

    y = mainDQN.predict(states)
    y[np.arange(len(X)), actions_id] = Q_target

    #print("Q_target:",Q_target)
    #print("screenQ_target:",screenQ_target)
    #print("minimapQ_target:",minimapQ_target)
    #print("np.arange(len(X)):",np.arange(len(X)), "len:",len(np.arange(len(X))))
    #print("actions_id:",actions_id, "len:",len(actions_id))
    #print("actions_args:",actions_arg, "len:", len(actions_arg))
    yscreen = mainScreenDQN.predict(states)
    yminimap = mainMinimapDQN.predict(states)
    for i in range(len(X)):
        if actions_arg[i] != -1:
            yscreen[i, actions_arg[i]] = screenQ_target[i]
            yminimap[i, actions_arg[i]] = minimapQ_target[i]
    mainScreenDQN.update(X, yscreen)
    mainMinimapDQN.update(X, yminimap)

    # Train our network using target and predicted Q values on each episode
    return mainDQN.update(X, y)


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
def run_loop(agents, env, sess, e, DQNlist, copy_list, max_frames=0):
    mainDQN = DQNlist[0]
    targetDQN = DQNlist[1]
    mainScreenDQN = DQNlist[2]
    targetScreenDQN = DQNlist[3]
    mainMinimapDQN = DQNlist[4]
    targetMinimapDQN = DQNlist[5]
    copy_ops = copy_list[0]
    copy_ops_screen = copy_list[1]
    copy_ops_minimap = copy_list[2]

    total_frames = 0
    start_time = time.time()

    action_spec = env.action_spec()
    observation_spec = env.observation_spec()
    for agent in agents:
        agent.setup(observation_spec, action_spec)

    timesteps = env.reset()
    state = minerva_agent.State(timesteps[0].observation)
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
            next_state = minerva_agent.State(timesteps[0].observation)
            reward = timesteps[0].reward
            done = timesteps[0].last()

            if done:
                break

            stored_buffer.append( (state, actions[0].function, actions[0].arguments, reward, next_state, done) )

            if len(stored_buffer) > batch_size:
                minibatch = random.sample(stored_buffer, batch_size)
                loss, _ = batch_train(env, DQNlist, minibatch)

            if step_count % update_frequency == 0:
                sess.run(copy_ops)
                sess.run(copy_ops_screen)
                sess.run(copy_ops_minimap)

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
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        mainDQN = dqn.DQN(sess, input_size, output_size, name="main")
        targetDQN = dqn.DQN(sess, input_size, output_size, name="target")
        mainScreenDQN = dqn.DQN(sess, input_size, screen_size, name="mainScreen")
        targetScreenDQN = dqn.DQN(sess, input_size, screen_size, name="targetScreen")
        mainMinimapDQN = dqn.DQN(sess, input_size, minimap_size, name="mainMinimap")
        targetMinimapDQN = dqn.DQN(sess, input_size, minimap_size, name="targetMinimap")
        sess.run(init)

        copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")
        copy_ops_screen = get_copy_var_ops(dest_scope_name="targetScreen", src_scope_name="mainScreen")
        copy_ops_minimap = get_copy_var_ops(dest_scope_name="targetMinimap", src_scope_name="mainMinimap")
        sess.run(copy_ops)
        sess.run(copy_ops_screen)
        sess.run(copy_ops_minimap)

        DQNlist = [mainDQN, targetDQN, mainScreenDQN, targetScreenDQN, mainMinimapDQN, targetScreenDQN]
        copy_list = [copy_ops, copy_ops_screen, copy_ops_minimap]

        for episode in range(num_episodes):
            e = 1.0 / ((episode / 50) + 10) # decaying exploration rate
            with sc2_env.SC2Env(
                    "Odyssey",
                    agent_race=myrace,
                    bot_race=botrace,
                    difficulty="1",
                    visualize=visualize) as env:
                agent = minerva_agent.MinervaAgent(DQNlist)
                run_result = run_loop([agent], env, sess, e, DQNlist, copy_list)
                reward = run_result[0].reward
                if reward > 0:
                    env.save_replay("victory/")

            print("%d'th game result :"%(episode+1),reward)
            saver.save(sess, "sessionSave")

argv = FLAGS(sys.argv)
sys.exit(_main(argv))
