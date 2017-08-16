import numpy as np
import tensorflow as tf

class DQN:

    def __init__(self, session: tf.Session, screen_size: int, minimap_size: int, output_size: int, name: str="main") -> None:
        """DQN Agent can
        1) Build network
        2) Predict Q_value given state
        3) Train parameters
        Args:
            session (tf.Session): Tensorflow session
            screen_size : screen width pixel size, default=64
            minimap_size : minimap width pixel size, default=64
            output_size (int): Number of discrete actions
            name (str, optional): TF Graph will be built under this name scope
        """
        self.session = session
        self.output_size = output_size
        self.screen_size = screen_size
        self.minimap_size = minimap_size
        self.net_name = name
        self._build_network()

    def _build_network(self, l_rate=0.001) -> None:
        """DQN Network architecture (simple MLP)
        Args:
            h_size (int, optional): Hidden layer dimension
            l_rate (float, optional): Learning rate
        """
        with tf.variable_scope(self.net_name):
            self.session.run(tf.global_variables_initializer())

            self._X_minimap = tf.placeholder(tf.float32, [None, 7, self.minimap_size, self.minimap_size], name="x_minimap")
            self._X_screen = tf.placeholder(tf.float32, [None, 13, self.screen_size, self.screen_size], name="x_screen")
            #self._X_select = tf.placeholder(tf.float32, [None, 1, 7], name="x_select")
            self._X_player = tf.placeholder(tf.float32, [None, 11], name="x_player")
            self._X_control_group = tf.placeholder(tf.float32, [None, 10, 2], name="x_control_group")
            self._X_score = tf.placeholder(tf.float32, [None, 13], name="x_score")
            _X_minimap = tf.transpose(self._X_minimap, perm=[0,2,3,1])
            _X_screen = tf.transpose(self._X_screen, perm=[0,2,3,1])

            W1_minimap = tf.Variable(tf.random_normal([3,3,7,13],stddev=0.1))
            L1_minimap = tf.nn.conv2d(_X_minimap, W1_minimap, strides=[1,1,1,1], padding="SAME")
            L1_minimap = tf.nn.relu(L1_minimap)
            L1_minimap = tf.nn.max_pool(L1_minimap, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

            W2_minimap = tf.Variable(tf.random_normal([3,3,13,13],stddev=0.1))
            L2_minimap = tf.nn.conv2d(L1_minimap, W2_minimap, strides=[1,1,1,1], padding="SAME")
            L2_minimap = tf.nn.relu(L2_minimap)
            L2_minimap = tf.nn.max_pool(L2_minimap, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
            # for default, L2_minimap shape : [-1, 16, 16, 12]

            W1_screen = tf.Variable(tf.random_normal([3,3,13,16],stddev=0.1))
            L1_screen = tf.nn.conv2d(_X_screen, W1_screen, strides=[1,1,1,1], padding="SAME")
            L1_screen = tf.nn.relu(L1_screen)
            L1_screen = tf.nn.max_pool(L1_screen, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

            W2_screen = tf.Variable(tf.random_normal([3,3,16,16],stddev=0.1))
            L2_screen = tf.nn.conv2d(L1_screen, W2_screen, strides=[1,1,1,1], padding="SAME")
            L2_screen = tf.nn.relu(L2_screen)
            L2_screen = tf.nn.max_pool(L2_screen, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
            # for default, L2_screen shape : [-1, 16, 16, 16]

            W1_player = tf.Variable(tf.random_normal([11, 256],stddev=0.1))
            L1_player = tf.matmul(self._X_player, W1_player)
            L1_player = tf.nn.relu(L1_player)
            L1_player = tf.reshape(L1_player,[-1, 16,16,1])

            #_X_select = tf.reshape(self._X_select,[-1,7])
            #W1_select = tf.Variable(tf.random_normal([7, 256],stddev=0.1))
            #L1_select = tf.matmul(_X_select, W1_select)
            #L1_select = tf.nn.relu(L1_select)
            #L1_select = tf.reshape(L1_select,[-1, 16,16,1])

            _X_control = tf.reshape(self._X_control_group,[-1,20])
            W1_control = tf.Variable(tf.random_normal([20, 256],stddev=0.1))
            L1_control = tf.matmul(_X_control, W1_control)
            L1_control = tf.nn.relu(L1_control)
            L1_control = tf.reshape(L1_control,[-1, 16,16,1])

            W1_score = tf.Variable(tf.random_normal([13, 256],stddev=0.1))
            L1_score = tf.matmul(self._X_score, W1_score)
            L1_score = tf.nn.relu(L1_score)
            L1_score = tf.reshape(L1_score,[-1, 16,16,1])

            # for default, _X_State shape : [-1, 16, 16, 32]
            _X_State = tf.concat([L2_minimap, L2_screen, L1_player, L1_control, L1_score], axis=-1)

            # *_ID : nets for classifying action_id (e.g. move_camera etc)
            W1_ID = tf.Variable(tf.random_normal([3,3,32,32], stddev=0.1), name="W1_ID")
            L1_ID = tf.nn.conv2d(_X_State, W1_ID, strides=[1,1,1,1], padding="SAME")
            L1_ID = tf.nn.relu(L1_ID)
            L1_ID = tf.nn.max_pool(L1_ID, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
            # L1_ID shape : [-1, 8, 8, 32]

            W2_ID = tf.Variable(tf.random_normal([3,3,32,64], stddev=0.1), name="W2_ID")
            L2_ID = tf.nn.conv2d(L1_ID, W2_ID, strides=[1,1,1,1], padding="SAME")
            L2_ID = tf.nn.relu(L2_ID)
            L2_ID = tf.nn.max_pool(L2_ID, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
            # L2_ID shape : [-1, 4, 4, 64]
            L2_ID = tf.reshape(L2_ID,[-1, 1024])

            W3_ID = tf.Variable(tf.random_normal([1024, self.output_size]))
            self._Qpred = tf.matmul(L2_ID, W3_ID)

            W1_spatial = tf.Variable(tf.random_normal([3,3,32,32],stddev=0.1), name="W1_spatial")
            L1_spatial = tf.nn.conv2d(_X_State, W1_spatial, strides=[1,1,1,1], padding="SAME")
            L1_spatial = tf.nn.relu(L1_spatial)
            L1_spatial = tf.nn.max_pool(L1_spatial, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
            # L1_spatial shape : [-1, 8, 8, 32]

            W2_spatial = tf.Variable(tf.random_normal([3,3,32,64],stddev=0.1), name="W2_spatial")
            L2_spatial = tf.nn.conv2d(L1_spatial, W2_spatial, strides=[1,1,1,1], padding="SAME")
            L2_spatial = tf.nn.relu(L2_spatial)
            L2_spatial = tf.nn.max_pool(L2_spatial, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
            # L2_spatial shape : [-1, 4, 4, 64]
            L2_spatial = tf.reshape(L2_spatial, [-1,1024])

            W3_spatial = tf.Variable(tf.random_normal([1024, self.screen_size*self.screen_size]))
            self._spatial_Qpred = tf.matmul(L2_spatial, W3_spatial)

            self._Y = tf.placeholder(tf.float32, shape=[None, self.output_size])
            self._Y_spatial = tf.placeholder(tf.float32, shape=[None, self.screen_size*self.screen_size])
            _loss1 = tf.losses.mean_squared_error(self._Y, self._Qpred)
            _loss2 = tf.losses.mean_squared_error(self._Y_spatial, self._spatial_Qpred)
            self._loss = _loss1 + _loss2

            optimizer = tf.train.AdamOptimizer(learning_rate=l_rate)
            self._train = optimizer.minimize(self._loss)

            self.saver = tf.train.Saver({
                'W1_minimap':W1_minimap,
                'W2_minimap':W2_minimap,
                'W1_screen':W1_screen,
                'W2_screen':W2_screen,
                'W1_player':W1_player,
                'W1_control':W1_control,
                'W1_score':W1_score,
                'W1_ID':W1_ID,
                'W2_ID':W2_ID,
                'W3_ID':W3_ID,
                'W1_spatial':W1_spatial,
                'W2_spatial':W2_spatial,
                'W3_spatial':W3_spatial
            })

            try:
                self.saver.restore(self.session, "saved/model")
                print("DQN : weight params are restored")
            except:
                print("DQN : no params were restored")

    def predict(self, state) -> np.ndarray:
        """Returns Q(s, a)
        Args:
            state (np.ndarray): State array, shape (n, input_dim)
        Returns:
            np.ndarray: Q value array, shape (n, output_dim)
        """
        #print("type:",type(state),"len:",len(state))
        #print("state type:",type(state[0]),"len:",len(state[0]))
        #print("state[0] type:",type(state[0][0]),"len:",len(state[0][0]))
        _minimap = np.vstack([x[0]['minimap'] for x in state])
        _screen = np.vstack([x[0]['screen'] for x in state])
        _control = np.vstack([x[0]['control_groups'] for x in state])
        _player = np.array([x[0]['player'] for x in state])
        _score = np.array([x[0]['score_cumulative'] for x in state])
        #_select = np.array([x[0]['single_select'] for x in state])

        _minimap = np.reshape(_minimap, [-1, 7, self.minimap_size, self.minimap_size])
        _screen = np.reshape(_screen, [-1, 13, self.screen_size, self.screen_size])
        _control = np.reshape(_control, [-1, 10, 2])
        _player = np.reshape(_player, [-1, 11])
        _score = np.reshape(_score, [-1, 13])
        feed = {
            self._X_minimap: _minimap,
            self._X_screen: _screen,
            self._X_control_group: _control,
            self._X_player: _player,
            self._X_score: _score
        }
        return self.session.run(self._Qpred, feed_dict=feed)

    def predictSpatial(self, state):
        _minimap = np.vstack([x[0]['minimap'] for x in state])
        _screen = np.vstack([x[0]['screen'] for x in state])
        _control = np.vstack([x[0]['control_groups'] for x in state])
        _player = np.array([x[0]['player'] for x in state])
        _score = np.array([x[0]['score_cumulative'] for x in state])
        #_select = np.array([x[0]['single_select'] for x in state])

        _minimap = np.reshape(_minimap, [-1, 7, self.minimap_size, self.minimap_size])
        _screen = np.reshape(_screen, [-1, 13, self.screen_size, self.screen_size])
        _control = np.reshape(_control, [-1, 10, 2])
        _player = np.reshape(_player, [-1, 11])
        _score = np.reshape(_score, [-1, 13])
        feed = {
            self._X_minimap: _minimap,
            self._X_screen: _screen,
            self._X_control_group: _control,
            self._X_player: _player,
            self._X_score: _score
        }
        return self.session.run(self._spatial_Qpred, feed_dict=feed)

    def update(self, state, y_stack, y_spatial) -> list:
        """Performs updates on given X and y and returns a result
        Args:
            x_stack (np.ndarray): State array, shape (n, input_dim)
            y_stack (np.ndarray): Target Q array, shape (n, output_dim)
        Returns:
            list: First element is loss, second element is a result from train step

        """
        _minimap = np.vstack([x[0]['minimap'] for x in state])
        _screen = np.vstack([x[0]['screen'] for x in state])
        _control = np.vstack([x[0]['control_groups'] for x in state])
        _player = np.array([x[0]['player'] for x in state])
        _score = np.array([x[0]['score_cumulative'] for x in state])
        #_select = np.array([x[0]['single_select'] for x in state])

        _minimap = np.reshape(_minimap, [-1, 7, self.minimap_size, self.minimap_size])
        _screen = np.reshape(_screen, [-1, 13, self.screen_size, self.screen_size])
        _control = np.reshape(_control, [-1, 10, 2])
        _player = np.reshape(_player, [-1, 11])
        _score = np.reshape(_score, [-1, 13])
        feed = {
            self._X_minimap: _minimap,
            self._X_screen: _screen,
            self._X_control_group: _control,
            self._X_player: _player,
            self._X_score: _score,
            self._Y: y_stack,
            self._Y_spatial: y_spatial
        }
        return self.session.run([self._loss, self._train], feed)

    def saveWeight(self):
        self.saver.save(self.session, 'saved/model')
