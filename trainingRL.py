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
learning_rate = 0.1
discount = 0.99
num_episodes = 100
batch_size = 64
max_buffer_size = 50000
update_frequency = 16

visualize = False
myrace = "P"
botrace = "T"

def batch_train(mainDQN: dqn.DQN, targetDQN: dqn.DQN, train_batch: list) -> float:
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
    actions = np.array([x[1] for x in train_batch])
    rewards = np.array([x[2] for x in train_batch])
    next_states = np.vstack([x[3] for x in train_batch])
    done = np.array([x[4] for x in train_batch])

    X = states

    Q_target = rewards + DISCOUNT_RATE * np.max(targetDQN.predict(next_states), axis=1) * ~done

    y = mainDQN.predict(states)
    y[np.arange(len(X)), actions] = Q_target

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
def run_loop(agents, env, sess, e, mainDQN, targetDQN, copy_ops, max_frames=0):
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
                actions = [agent.step(timestep, mainDQN.predict(state))
                       for agent, timestep in zip(agents, timesteps)]

            if max_frames and total_frames >= max_frames:
                return timesteps

            timesteps = env.step(actions)
            next_state = minerva_agent.State(timesteps[0].observation)
            reward = timesteps[0].reward
            done = timesteps[0].last()

            if done:
                break

            stored_buffer.append( (state, actions[0], reward, next_state, done) )

            if len(stored_buffer) > batch_size:
                minibatch = random.sample(stored_buffer, batch_size)
                loss, _ = batch_train(mainDQN, targetDQN, minibatch)

            if step_count % update_frequency == 0:
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

XT = tf.zeros([1])

def _main(unused_argv):
    store_buffer = deque(maxlen=max_buffer_size)

    init = tf.global_variables_initializer()
    #init = tf.initialize_all_variables()
    with tf.Session() as sess:
        mainDQN = dqn.DQN(sess, input_size, output_size, name="main")
        targetDQN = dqn.DQN(sess, input_size, output_size, name="target")
        sess.run(init)

        copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")
        #sess.run(copy_ops)

        for episode in range(num_episodes):
            e = 1.0 / ((episode / 50) + 10) # decaying exploration rate
            with sc2_env.SC2Env(
                    "Odyssey",
                    agent_race=myrace,
                    bot_race=botrace,
                    difficulty="1",
                    visualize=visualize) as env:
                agent = minerva_agent.MinervaAgent()
                run_result = run_loop([agent], env, sess, e, mainDQN, targetDQN, copy_ops) # no max_frame set : infinite game time

            reward = run_result[0].reward
            print("%d'th game result :"%(episode+1),reward)

argv = FLAGS(sys.argv)
sys.exit(_main(argv))
