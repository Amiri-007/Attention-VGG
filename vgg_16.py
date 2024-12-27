import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

class VGG16Base:
    """
    VGG16 model with placeholder-based attention in each conv layer.
    """
    def __init__(self, imgs, labs=None, weights=None, sess=None):
        self.imgs = imgs
        self.convlayers()
        self.fc_layers()

        self.guess = tf.round(tf.nn.sigmoid(self.fc3l))

        if labs is not None:
            xent = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fc3l, labels=tf.cast(labs, tf.float32))
            l2_loss = tf.nn.l2_loss(self.fc3w)
            self.cross_entropy = tf.reduce_mean(xent) + 0.01 * l2_loss
            self.train_step = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-2).minimize(self.cross_entropy)

        if weights is not None and sess is not None:
            self.load_weights(weights, sess)
    
    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file, allow_pickle=True)
        keys = sorted(weights.keys())
        keys = keys[0:-2]  # ignore last two
        sess.run(tf.compat.v1.global_variables_initializer())
        for i, k in enumerate(keys):
            print(i, k, np.shape(weights[k]))
            sess.run(self.parameters[i].assign(weights[k]))

    def get_all_layers(self):
        return [
            self.smean1_1, self.smean1_2,
            self.smean2_1, self.smean2_2,
            self.smean3_1, self.smean3_2, self.smean3_3,
            self.smean4_1, self.smean4_2, self.smean4_3,
            self.smean5_1, self.smean5_2, self.smean5_3
        ]

    def get_attention_placeholders(self):
        return [
            self.a11, self.a12,
            self.a21, self.a22,
            self.a31, self.a32, self.a33,
            self.a41, self.a42, self.a43,
            self.a51, self.a52, self.a53
        ]

    def convlayers(self):
        self.parameters = []

        with tf.name_scope('preprocess'):
            mean = tf.constant([123.68,116.779,103.939], dtype=tf.float32, shape=[1,1,1,3])
            images = self.imgs - mean

        # block1: conv1_1, conv1_2
        with tf.name_scope('conv1_1'):
            self.a11 = tf.compat.v1.placeholder(tf.float32, [224,224,64], name='a11')
            k11 = tf.Variable(tf.truncated_normal([3,3,3,64], stddev=1e-1), trainable=False, name='w11')
            b11 = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=False, name='b11')
            c11 = tf.nn.conv2d(images, k11, [1,1,1,1], padding='SAME')
            c11_out = tf.nn.bias_add(c11, b11)
            self.conv1_1 = tf.multiply(tf.nn.relu(c11_out), self.a11)
            self.parameters += [k11,b11]
            self.smean1_1 = tf.reduce_mean(self.conv1_1, [1,2])
            print('c11', self.conv1_1.get_shape().as_list())

        with tf.name_scope('conv1_2'):
            self.a12 = tf.compat.v1.placeholder(tf.float32, [224,224,64], name='a12')
            k12 = tf.Variable(tf.truncated_normal([3,3,64,64], stddev=1e-1), trainable=False, name='w12')
            b12 = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=False, name='b12')
            c12 = tf.nn.conv2d(self.conv1_1, k12, [1,1,1,1], padding='SAME')
            c12_out = tf.nn.bias_add(c12, b12)
            self.conv1_2 = tf.multiply(tf.nn.relu(c12_out), self.a12)
            self.parameters += [k12,b12]
            self.smean1_2 = tf.reduce_mean(self.conv1_2, [1,2])
            print('c12', self.conv1_2.get_shape().as_list())

        self.pool1 = tf.nn.max_pool2d(self.conv1_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

        # block2: conv2_1, conv2_2
        with tf.name_scope('conv2_1'):
            self.a21 = tf.compat.v1.placeholder(tf.float32, [112,112,128], name='a21')
            k21 = tf.Variable(tf.truncated_normal([3,3,64,128], stddev=1e-1), trainable=False, name='w21')
            b21 = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=False, name='b21')
            c21 = tf.nn.conv2d(self.pool1, k21, [1,1,1,1], padding='SAME')
            c21_out = tf.nn.bias_add(c21, b21)
            self.conv2_1 = tf.multiply(tf.nn.relu(c21_out), self.a21)
            self.parameters += [k21,b21]
            self.smean2_1 = tf.reduce_mean(self.conv2_1, [1,2])
            print('c21', self.conv2_1.get_shape().as_list())

        with tf.name_scope('conv2_2'):
            self.a22 = tf.compat.v1.placeholder(tf.float32, [112,112,128], name='a22')
            k22 = tf.Variable(tf.truncated_normal([3,3,128,128], stddev=1e-1), trainable=False, name='w22')
            b22 = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=False, name='b22')
            c22 = tf.nn.conv2d(self.conv2_1, k22, [1,1,1,1], padding='SAME')
            c22_out = tf.nn.bias_add(c22, b22)
            self.conv2_2 = tf.multiply(tf.nn.relu(c22_out), self.a22)
            self.parameters += [k22,b22]
            self.smean2_2 = tf.reduce_mean(self.conv2_2, [1,2])
            print('c22', self.conv2_2.get_shape().as_list())

        self.pool2 = tf.nn.max_pool2d(self.conv2_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

        # block3: conv3_1, conv3_2, conv3_3
        with tf.name_scope('conv3_1'):
            self.a31 = tf.compat.v1.placeholder(tf.float32, [56,56,256], name='a31')
            k31 = tf.Variable(tf.truncated_normal([3,3,128,256], stddev=1e-1), trainable=False, name='w31')
            b31 = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=False, name='b31')
            c31 = tf.nn.conv2d(self.pool2, k31, [1,1,1,1], padding='SAME')
            c31_out = tf.nn.bias_add(c31, b31)
            self.conv3_1 = tf.multiply(tf.nn.relu(c31_out), self.a31)
            self.parameters += [k31,b31]
            self.smean3_1 = tf.reduce_mean(self.conv3_1, [1,2])
            print('c31', self.conv3_1.get_shape().as_list())

        with tf.name_scope('conv3_2'):
            self.a32 = tf.compat.v1.placeholder(tf.float32, [56,56,256], name='a32')
            k32 = tf.Variable(tf.truncated_normal([3,3,256,256], stddev=1e-1), trainable=False, name='w32')
            b32 = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=False, name='b32')
            c32 = tf.nn.conv2d(self.conv3_1, k32, [1,1,1,1], padding='SAME')
            c32_out = tf.nn.bias_add(c32, b32)
            self.conv3_2 = tf.multiply(tf.nn.relu(c32_out), self.a32)
            self.parameters += [k32,b32]
            self.smean3_2 = tf.reduce_mean(self.conv3_2, [1,2])
            print('c32', self.conv3_2.get_shape().as_list())

        with tf.name_scope('conv3_3'):
            self.a33 = tf.compat.v1.placeholder(tf.float32, [56,56,256], name='a33')
            k33 = tf.Variable(tf.truncated_normal([3,3,256,256], stddev=1e-1), trainable=False, name='w33')
            b33 = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=False, name='b33')
            c33 = tf.nn.conv2d(self.conv3_2, k33, [1,1,1,1], padding='SAME')
            c33_out = tf.nn.bias_add(c33, b33)
            self.conv3_3 = tf.multiply(tf.nn.relu(c33_out), self.a33)
            self.parameters += [k33,b33]
            self.smean3_3 = tf.reduce_mean(self.conv3_3, [1,2])
            print('c33', self.conv3_3.get_shape().as_list())

        self.pool3 = tf.nn.max_pool2d(self.conv3_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

        # block4: conv4_1, conv4_2, conv4_3
        with tf.name_scope('conv4_1'):
            self.a41 = tf.compat.v1.placeholder(tf.float32, [28,28,512], name='a41')
            k41 = tf.Variable(tf.truncated_normal([3,3,256,512], stddev=1e-1), trainable=False, name='w41')
            b41 = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=False, name='b41')
            c41 = tf.nn.conv2d(self.pool3, k41, [1,1,1,1], padding='SAME')
            c41_out = tf.nn.bias_add(c41, b41)
            self.conv4_1 = tf.multiply(tf.nn.relu(c41_out), self.a41)
            self.parameters += [k41,b41]
            self.smean4_1 = tf.reduce_mean(self.conv4_1, [1,2])
            print('c41', self.conv4_1.get_shape().as_list())

        with tf.name_scope('conv4_2'):
            self.a42 = tf.compat.v1.placeholder(tf.float32, [28,28,512], name='a42')
            k42 = tf.Variable(tf.truncated_normal([3,3,512,512], stddev=1e-1), trainable=False, name='w42')
            b42 = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=False, name='b42')
            c42 = tf.nn.conv2d(self.conv4_1, k42, [1,1,1,1], padding='SAME')
            c42_out = tf.nn.bias_add(c42, b42)
            self.conv4_2 = tf.multiply(tf.nn.relu(c42_out), self.a42)
            self.parameters += [k42,b42]
            self.smean4_2 = tf.reduce_mean(self.conv4_2, [1,2])
            print('c42', self.conv4_2.get_shape().as_list())

        with tf.name_scope('conv4_3'):
            self.a43 = tf.compat.v1.placeholder(tf.float32, [28,28,512], name='a43')
            k43 = tf.Variable(tf.truncated_normal([3,3,512,512], stddev=1e-1), trainable=False, name='w43')
            b43 = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=False, name='b43')
            c43 = tf.nn.conv2d(self.conv4_2, k43, [1,1,1,1], padding='SAME')
            c43_out = tf.nn.bias_add(c43, b43)
            self.conv4_3 = tf.multiply(tf.nn.relu(c43_out), self.a43)
            self.parameters += [k43,b43]
            self.smean4_3 = tf.reduce_mean(self.conv4_3, [1,2])
            print('c43', self.conv4_3.get_shape().as_list())

        self.pool4 = tf.nn.max_pool2d(self.conv4_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

        # block5: conv5_1, conv5_2, conv5_3
        with tf.name_scope('conv5_1'):
            self.a51 = tf.compat.v1.placeholder(tf.float32, [14,14,512], name='a51')
            k51 = tf.Variable(tf.truncated_normal([3,3,512,512], stddev=1e-1), trainable=False, name='w51')
            b51 = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=False, name='b51')
            c51 = tf.nn.conv2d(self.pool4, k51, [1,1,1,1], padding='SAME')
            c51_out = tf.nn.bias_add(c51, b51)
            self.conv5_1 = tf.multiply(tf.nn.relu(c51_out), self.a51)
            self.parameters += [k51,b51]
            self.smean5_1 = tf.reduce_mean(self.conv5_1, [1,2])
            print('c51', self.conv5_1.get_shape().as_list())

        with tf.name_scope('conv5_2'):
            self.a52 = tf.compat.v1.placeholder(tf.float32, [14,14,512], name='a52')
            k52 = tf.Variable(tf.truncated_normal([3,3,512,512], stddev=1e-1), trainable=False, name='w52')
            b52 = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=False, name='b52')
            c52 = tf.nn.conv2d(self.conv5_1, k52, [1,1,1,1], padding='SAME')
            c52_out = tf.nn.bias_add(c52, b52)
            self.conv5_2 = tf.multiply(tf.nn.relu(c52_out), self.a52)
            self.parameters += [k52,b52]
            self.smean5_2 = tf.reduce_mean(self.conv5_2, [1,2])
            print('c52', self.conv5_2.get_shape().as_list())

        with tf.name_scope('conv5_3'):
            self.a53 = tf.compat.v1.placeholder(tf.float32, [14,14,512], name='a53')
            k53 = tf.Variable(tf.truncated_normal([3,3,512,512], stddev=1e-1), trainable=False, name='w53')
            b53 = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=False, name='b53')
            c53 = tf.nn.conv2d(self.conv5_2, k53, [1,1,1,1], padding='SAME')
            c53_out = tf.nn.bias_add(c53, b53)
            self.conv5_3 = tf.multiply(tf.nn.relu(c53_out), self.a53)
            self.parameters += [k53,b53]
            self.smean5_3 = tf.reduce_mean(self.conv5_3, [1,2])
            print('c53', self.conv5_3.get_shape().as_list())

        self.pool5 = tf.nn.max_pool2d(self.conv5_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    def fc_layers(self):
        shape = int(np.prod(self.pool5.get_shape()[1:]))

        fc1w = tf.Variable(tf.truncated_normal([shape, 4096], dtype=tf.float32, stddev=1e-1),
                           trainable=False, name='fc1w')
        fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                           trainable=False, name='fc1b')
        pool5_flat = tf.reshape(self.pool5, [-1, shape])
        fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
        self.fc1 = tf.nn.relu(fc1l)
        self.parameters += [fc1w, fc1b]

        fc2w = tf.Variable(tf.truncated_normal([4096,4096], dtype=tf.float32, stddev=1e-1),
                           trainable=False, name='fc2w')
        fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                           trainable=False, name='fc2b')
        fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
        self.fc2 = tf.nn.relu(fc2l)
        self.parameters += [fc2w, fc2b]

        self.fc3w = tf.Variable(tf.truncated_normal([4096,1], dtype=tf.float32, stddev=1e-1),
                                trainable=True, name='fc3w')
        self.fc3b = tf.Variable(tf.constant(1.0, shape=[1], dtype=tf.float32),
                                trainable=True, name='fc3b')
        self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, self.fc3w), self.fc3b)
        self.parameters += [self.fc3w, self.fc3b]
