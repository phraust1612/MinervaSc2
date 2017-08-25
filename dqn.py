import numpy as np
import tensorflow as tf

class DQN:

    def __init__(self, session: tf.Session, screen_size: int, minimap_size: int, output_size: int, learning_rate:int, name: str="main") -> None:
        """DQN Agent can
        1) Build network
        2) Predict Q_value given state
        3) Train parameters
        Args:
            session (tf.Session): Tensorflow session
            screen_size : screen width pixel size, default=64
            minimap_size : minimap width pixel size, default=64
            output_size (int): Number of discrete actions
            learning_rate (int): do I need any more explanation?
            name (str, optional): TF Graph will be built under this name scope
        """
        self.session = session
        self.output_size = output_size
        self.screen_size = screen_size
        self.minimap_size = minimap_size
        self.net_name = name
        self.l_rate = learning_rate
        self._build_network()

    def _build_network(self) -> None:
        """DQN Network architecture (FullyConv : check out for DeepMind's sc2le paper)
        """
        with tf.variable_scope(self.net_name):
            self.session.run(tf.global_variables_initializer())

            self._X_minimap = tf.placeholder(tf.float32, [None, 7, self.minimap_size, self.minimap_size], name="x_minimap")
            self._X_screen = tf.placeholder(tf.float32, [None, 13, self.screen_size, self.screen_size], name="x_screen")
            self._X_select = tf.placeholder(tf.float32, [None, 1, 7], name="x_select")
            self._X_player = tf.placeholder(tf.float32, [None, 11], name="x_player")
            self._X_control_group = tf.placeholder(tf.float32, [None, 10, 2], name="x_control_group")
            self._X_score = tf.placeholder(tf.float32, [None, 13], name="x_score")
            _X_minimap = tf.transpose(self._X_minimap, perm=[0,2,3,1])
            _X_screen = tf.transpose(self._X_screen, perm=[0,2,3,1])

            W1_minimap = tf.Variable(tf.random_normal([3,3,7,12],stddev=0.1),name='W1_minimap')
            L1_minimap = tf.nn.conv2d(_X_minimap, W1_minimap, strides=[1,1,1,1], padding="SAME")
            L1_minimap = tf.nn.relu(L1_minimap)
            L1_minimap = tf.nn.max_pool(L1_minimap, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

            W2_minimap = tf.Variable(tf.random_normal([3,3,12,12],stddev=0.1),name='W2_minimap')
            L2_minimap = tf.nn.conv2d(L1_minimap, W2_minimap, strides=[1,1,1,1], padding="SAME")
            L2_minimap = tf.nn.relu(L2_minimap)
            L2_minimap = tf.nn.max_pool(L2_minimap, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
            # for default, L2_minimap shape : [-1, 16, 16, 12]

            W1_screen = tf.Variable(tf.random_normal([3,3,13,16],stddev=0.1),name='W1_screen')
            L1_screen = tf.nn.conv2d(_X_screen, W1_screen, strides=[1,1,1,1], padding="SAME")
            L1_screen = tf.nn.relu(L1_screen)
            L1_screen = tf.nn.max_pool(L1_screen, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

            W2_screen = tf.Variable(tf.random_normal([3,3,16,16],stddev=0.1),name='W2_screen')
            L2_screen = tf.nn.conv2d(L1_screen, W2_screen, strides=[1,1,1,1], padding="SAME")
            L2_screen = tf.nn.relu(L2_screen)
            L2_screen = tf.nn.max_pool(L2_screen, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
            # for default, L2_screen shape : [-1, 16, 16, 16]

            W1_player = tf.Variable(tf.random_normal([11, 256],stddev=0.1), name="W1_player")
            L1_player = tf.matmul(self._X_player, W1_player)
            L1_player = tf.nn.relu(L1_player)
            L1_player = tf.reshape(L1_player,[-1, 16,16,1])

            _X_select = tf.reshape(self._X_select,[-1,7])
            W1_select = tf.Variable(tf.random_normal([7, 256],stddev=0.1),name='W1_select')
            L1_select = tf.matmul(_X_select, W1_select)
            L1_select = tf.nn.relu(L1_select)
            L1_select = tf.reshape(L1_select,[-1, 16,16,1])

            _X_control = tf.reshape(self._X_control_group,[-1,20])
            W1_control = tf.Variable(tf.random_normal([20, 256],stddev=0.1),name='W1_control')
            L1_control = tf.matmul(_X_control, W1_control)
            L1_control = tf.nn.relu(L1_control)
            L1_control = tf.reshape(L1_control,[-1, 16,16,1])

            W1_score = tf.Variable(tf.random_normal([13, 256],stddev=0.1),name='W1_score')
            L1_score = tf.matmul(self._X_score, W1_score)
            L1_score = tf.nn.relu(L1_score)
            L1_score = tf.reshape(L1_score,[-1, 16,16,1])

            # for default, _X_State shape : [-1, 16, 16, 32]
            _X_State = tf.concat([L2_minimap, L2_screen, L1_player, L1_select, L1_control, L1_score], axis=-1)

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

            W3_ID = tf.Variable(tf.random_normal([1024, self.output_size]),name='W3_ID')
            self._Qpred = tf.matmul(L2_ID, W3_ID)

            W_screen_policy = tf.Variable(tf.random_normal([1024, self.screen_size*self.screen_size]),name='W_screen_policy')
            W_minimap_policy = tf.Variable(tf.random_normal([1024, self.minimap_size*self.minimap_size]),name="W_minimap_policy")
            W_screen2_policy = tf.Variable(tf.random_normal([1024, self.screen_size*self.screen_size]),name="W_screen2_policy")
            self._screen_policy_Qpred = tf.matmul(L2_ID, W_screen_policy)
            self._minimap_policy_Qpred = tf.matmul(L2_ID, W_minimap_policy)
            self._screen2_policy_Qpred = tf.matmul(L2_ID, W_screen2_policy)

            W_nonspatial3 = tf.Variable(tf.random_normal([1024, 2]),name='W_nonspatial3')
            W_nonspatial4 = tf.Variable(tf.random_normal([1024, 5]),name='W_nonspatial4')
            W_nonspatial5 = tf.Variable(tf.random_normal([1024, 10]),name='W_nonspatial5')
            W_nonspatial6 = tf.Variable(tf.random_normal([1024, 4]),name='W_nonspatial6')
            W_nonspatial7 = tf.Variable(tf.random_normal([1024, 2]),name='W_nonspatial7')
            W_nonspatial8 = tf.Variable(tf.random_normal([1024, 4]),name='W_nonspatial8')
            W_nonspatial9 = tf.Variable(tf.random_normal([1024, 500]),name='W_nonspatial9')
            W_nonspatial10 = tf.Variable(tf.random_normal([1024, 4]),name='W_nonspatial10')
            W_nonspatial11 = tf.Variable(tf.random_normal([1024, 10]),name='W_nonspatial11')
            W_nonspatial12 = tf.Variable(tf.random_normal([1024, 500]),name='W_nonspatial12')
            self._nonspatial3_Qpred = tf.matmul(L2_ID, W_nonspatial3)
            self._nonspatial4_Qpred = tf.matmul(L2_ID, W_nonspatial4)
            self._nonspatial5_Qpred = tf.matmul(L2_ID, W_nonspatial5)
            self._nonspatial6_Qpred = tf.matmul(L2_ID, W_nonspatial6)
            self._nonspatial7_Qpred = tf.matmul(L2_ID, W_nonspatial7)
            self._nonspatial8_Qpred = tf.matmul(L2_ID, W_nonspatial8)
            self._nonspatial9_Qpred = tf.matmul(L2_ID, W_nonspatial9)
            self._nonspatial10_Qpred = tf.matmul(L2_ID, W_nonspatial10)
            self._nonspatial11_Qpred = tf.matmul(L2_ID, W_nonspatial11)
            self._nonspatial12_Qpred = tf.matmul(L2_ID, W_nonspatial12)

            self._Y = tf.placeholder(tf.float32, shape=[None, self.output_size])
            self._Y_screen = tf.placeholder(tf.float32, shape=[None, self.screen_size*self.screen_size])
            self._Y_minimap = tf.placeholder(tf.float32, shape=[None, self.minimap_size*self.minimap_size])
            self._Y_screen2 = tf.placeholder(tf.float32, shape=[None, self.screen_size*self.screen_size])
            self._Y_nonspatial3 = tf.placeholder(tf.float32, shape=[None, 2])
            self._Y_nonspatial4 = tf.placeholder(tf.float32, shape=[None, 5])
            self._Y_nonspatial5 = tf.placeholder(tf.float32, shape=[None, 10])
            self._Y_nonspatial6 = tf.placeholder(tf.float32, shape=[None, 4])
            self._Y_nonspatial7 = tf.placeholder(tf.float32, shape=[None, 2])
            self._Y_nonspatial8 = tf.placeholder(tf.float32, shape=[None, 4])
            self._Y_nonspatial9 = tf.placeholder(tf.float32, shape=[None, 500])
            self._Y_nonspatial10 = tf.placeholder(tf.float32, shape=[None, 4])
            self._Y_nonspatial11 = tf.placeholder(tf.float32, shape=[None, 10])
            self._Y_nonspatial12 = tf.placeholder(tf.float32, shape=[None, 500])

            _loss = tf.losses.mean_squared_error(self._Y, self._Qpred) * 10
            _loss += tf.losses.mean_squared_error(self._Y_screen, self._screen_policy_Qpred)
            _loss += tf.losses.mean_squared_error(self._Y_minimap, self._minimap_policy_Qpred)
            _loss += tf.losses.mean_squared_error(self._Y_screen2, self._screen2_policy_Qpred)
            _loss += tf.losses.mean_squared_error(self._Y_nonspatial3, self._nonspatial3_Qpred)
            _loss += tf.losses.mean_squared_error(self._Y_nonspatial4, self._nonspatial4_Qpred)
            _loss += tf.losses.mean_squared_error(self._Y_nonspatial5, self._nonspatial5_Qpred)
            _loss += tf.losses.mean_squared_error(self._Y_nonspatial6, self._nonspatial6_Qpred)
            _loss += tf.losses.mean_squared_error(self._Y_nonspatial7, self._nonspatial7_Qpred)
            _loss += tf.losses.mean_squared_error(self._Y_nonspatial8, self._nonspatial8_Qpred)
            _loss += tf.losses.mean_squared_error(self._Y_nonspatial9, self._nonspatial9_Qpred)
            _loss += tf.losses.mean_squared_error(self._Y_nonspatial10, self._nonspatial10_Qpred)
            _loss += tf.losses.mean_squared_error(self._Y_nonspatial11, self._nonspatial11_Qpred)
            _loss += tf.losses.mean_squared_error(self._Y_nonspatial12, self._nonspatial12_Qpred)
            self._loss = _loss

            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.l_rate)
            self._train = optimizer.minimize(self._loss)

            self.saver = tf.train.Saver({
                'W1_minimap':W1_minimap,
                'W2_minimap':W2_minimap,
                'W1_screen':W1_screen,
                'W2_screen':W2_screen,
                'W1_player':W1_player,
                'W1_select':W1_select,
                'W1_control':W1_control,
                'W1_score':W1_score,
                'W1_ID':W1_ID,
                'W2_ID':W2_ID,
                'W3_ID':W3_ID,
                'W_screen_policy':W_screen_policy,
                'W_minimap_policy':W_minimap_policy,
                'W_screen2_policy':W_screen2_policy,
                'W_nonspatial3':W_nonspatial3,
                'W_nonspatial4':W_nonspatial4,
                'W_nonspatial5':W_nonspatial5,
                'W_nonspatial6':W_nonspatial6,
                'W_nonspatial7':W_nonspatial7,
                'W_nonspatial8':W_nonspatial8,
                'W_nonspatial9':W_nonspatial9,
                'W_nonspatial10':W_nonspatial10,
                'W_nonspatial11':W_nonspatial11,
                'W_nonspatial12':W_nonspatial12
            })

            try:
                self.saver.restore(self.session, "saved/model")
                print("DQN : weight params are restored")
            except:
                print("DQN : no params were restored")

    def predict(self, state) -> np.ndarray:
        """Returns Q(s, a) <- here the Q only predicts for action id, not arguments
        Args:
            state (array): State array, shape (n, )
        Returns:
            np.ndarray: Q value array, shape (n, output_dim)
        """
        _minimap = np.vstack([x[0]['minimap'] for x in state])
        _screen = np.vstack([x[0]['screen'] for x in state])
        _control = np.vstack([x[0]['control_groups'] for x in state])
        _player = np.array([x[0]['player'] for x in state])
        _score = np.array([x[0]['score_cumulative'] for x in state])
        _select = np.array([x[0]['single_select'] for x in state])
        _multiselect = np.array([x[0]['multi_select'] for x in state])

        _minimap = np.reshape(_minimap, [-1, 7, self.minimap_size, self.minimap_size])
        _screen = np.reshape(_screen, [-1, 13, self.screen_size, self.screen_size])
        _control = np.reshape(_control, [-1, 10, 2])
        _player = np.reshape(_player, [-1, 11])
        _score = np.reshape(_score, [-1, 13])
        _select = np.reshape(_select, [-1, 1, 7])
        for i in range(len(_multiselect)):
            if _multiselect[i].shape[0] > 0:
                _select[i][0] = _multiselect[i][0]

        feed = {
            self._X_minimap: _minimap,
            self._X_screen: _screen,
            self._X_control_group: _control,
            self._X_player: _player,
            self._X_score: _score,
            self._X_select: _select
        }
        return self.session.run(self._Qpred, feed_dict=feed)

    def predictSpatial(self, state):
        """Returns spatial/nonspatial argument Q values
        """
        _minimap = np.vstack([x[0]['minimap'] for x in state])
        _screen = np.vstack([x[0]['screen'] for x in state])
        _control = np.vstack([x[0]['control_groups'] for x in state])
        _player = np.array([x[0]['player'] for x in state])
        _score = np.array([x[0]['score_cumulative'] for x in state])
        _select = np.array([x[0]['single_select'] for x in state])
        _multiselect = np.array([x[0]['multi_select'] for x in state])

        _minimap = np.reshape(_minimap, [-1, 7, self.minimap_size, self.minimap_size])
        _screen = np.reshape(_screen, [-1, 13, self.screen_size, self.screen_size])
        _control = np.reshape(_control, [-1, 10, 2])
        _player = np.reshape(_player, [-1, 11])
        _score = np.reshape(_score, [-1, 13])
        _select = np.reshape(_select, [-1, 1, 7])
        for i in range(len(_multiselect)):
            if _multiselect[i].shape[0] > 0:
                _select[i][0] = _multiselect[i][0]


        feed = {
            self._X_minimap: _minimap,
            self._X_screen: _screen,
            self._X_control_group: _control,
            self._X_player: _player,
            self._X_score: _score,
            self._X_select: _select
        }
        return self.session.run([
            self._screen_policy_Qpred,
            self._minimap_policy_Qpred,
            self._screen2_policy_Qpred,
            self._nonspatial3_Qpred,
            self._nonspatial4_Qpred,
            self._nonspatial5_Qpred,
            self._nonspatial6_Qpred,
            self._nonspatial7_Qpred,
            self._nonspatial8_Qpred,
            self._nonspatial9_Qpred,
            self._nonspatial10_Qpred,
            self._nonspatial11_Qpred,
            self._nonspatial12_Qpred], feed_dict=feed)

    def update(self, state, y_stack, y_spatial) -> list:
        """Performs updates on given X and y and returns a result
        Args:
            x_stack (array): State array, shape (n, )
            y_stack (array): Target action id Q array, shape (n, output_dim)
            y_spatial (array) : Target action argument Q array (13, n, )
        Returns:
            list: First element is loss, second element is a result from train step

        """
        _minimap = np.vstack([x[0]['minimap'] for x in state])
        _screen = np.vstack([x[0]['screen'] for x in state])
        _control = np.vstack([x[0]['control_groups'] for x in state])
        _player = np.array([x[0]['player'] for x in state])
        _score = np.array([x[0]['score_cumulative'] for x in state])
        _select = np.array([x[0]['single_select'] for x in state])
        _multiselect = np.array([x[0]['multi_select'] for x in state])

        _minimap = np.reshape(_minimap, [-1, 7, self.minimap_size, self.minimap_size])
        _screen = np.reshape(_screen, [-1, 13, self.screen_size, self.screen_size])
        _control = np.reshape(_control, [-1, 10, 2])
        _player = np.reshape(_player, [-1, 11])
        _score = np.reshape(_score, [-1, 13])
        _select = np.reshape(_select, [-1, 1, 7])
        for i in range(len(_multiselect)):
            if _multiselect[i].shape[0] > 0:
                _select[i][0] = _multiselect[i][0]


        feed = {
            self._X_minimap: _minimap,
            self._X_screen: _screen,
            self._X_control_group: _control,
            self._X_player: _player,
            self._X_score: _score,
            self._X_select: _select,
            self._Y: y_stack,
            self._Y_screen: y_spatial[0],
            self._Y_minimap: y_spatial[1],
            self._Y_screen2: y_spatial[2],
            self._Y_nonspatial3: y_spatial[3],
            self._Y_nonspatial4: y_spatial[4],
            self._Y_nonspatial5: y_spatial[5],
            self._Y_nonspatial6: y_spatial[6],
            self._Y_nonspatial7: y_spatial[7],
            self._Y_nonspatial8: y_spatial[8],
            self._Y_nonspatial9: y_spatial[9],
            self._Y_nonspatial10: y_spatial[10],
            self._Y_nonspatial11: y_spatial[11],
            self._Y_nonspatial12: y_spatial[12],
        }
        return self.session.run([self._loss, self._train], feed)

    def saveWeight(self):
        self.saver.save(self.session, 'saved/model')
