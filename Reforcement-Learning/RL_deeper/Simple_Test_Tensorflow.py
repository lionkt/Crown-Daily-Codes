import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

MAX_STEP = 1000


def linear_regression():
    # 代码段功能学习线性拟合的参数
    # 构造数据集
    x_data = np.random.rand(100).astype(np.float32)
    y_data = x_data * 0.1 + 0.3

    weight = tf.Variable(tf.random_uniform([1], -1, 1))
    # bias = tf.Variable(tf.zeros([1]))
    bias = tf.Variable(0.0)
    y = weight * x_data + bias

    loss = tf.reduce_mean(tf.square(y - y_data))  # 这里只是定义运算，而不是执行
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.4)  # 采用gradient-descent法优化
    train = optimizer.minimize(loss)
    init = tf.global_variables_initializer()  # 初始化对于variable的使用是必要的步骤

    # 训练的过程
    # 对于tf.Session()尽量用with语句块，这样能保证出了with块session就自动关闭。否则一直在内存里
    with tf.Session() as sess:
        sess.run(init)
        for i in range(MAX_STEP):
            sess.run(train)
            if i % 20 == 0:
                print('step: ', i, ', weight: ', sess.run(weight), ', bias: ', sess.run(bias))  # 这里必须要加sess.run()才能访问结果


def test_placeholder():
    # 在 Tensorflow 中需要定义 placeholder 的 type ，一般为 float32 形式
    input1 = tf.placeholder(tf.float32)
    input2 = tf.placeholder(tf.float32)
    output = tf.multiply(input1, input2)
    with tf.Session() as sess:
        res = sess.run(output, feed_dict={input1: [7.], input2: [2.]})
        print(res)


def linear_official_test():
    # tensorflow 官方test
    # train data
    x_data = np.array([1, 2, 3, 4])
    y_data = np.array([0., -1., -2., -3.])
    # input and output
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    # model parameters
    weights = tf.Variable([.0], dtype=tf.float32)
    bias = tf.Variable([.0], dtype=tf.float32)
    linear_model = weights * x + bias
    # loss function
    loss = tf.reduce_mean(tf.square(y - linear_model))
    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    # train mainloop
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(MAX_STEP):
            sess.run(train, {x: x_data, y: y_data})

        # evaluate
        # final_W, final_b = sess.run([weights, bias])  # variable作为变量，存储了训练的结果。因此该语句可正常运行
        # final_loss = sess.run(loss) # 由于涉及placeholder的语句不直接存储数据，需要指定输入的数据
        final_W, final_b, final_loss = sess.run([weights, bias, loss], {x: x_data, y: y_data})
        print("w:%s, b:%s" % (final_W, final_b))


# 给神经网络添加层
# 包含了对tensorboard的命名块测试，以及训练过程统计
def add_layer(inputs, in_size, out_size, activation_function=None, layer_name=None):
    my_name = ''
    if layer_name is None:
        my_name = 'layer'
    else:
        my_name = layer_name
    with tf.name_scope(my_name):
        with tf.name_scope('w'):
            weight = tf.Variable(tf.random_normal([in_size, out_size]))
            tf.summary.histogram(layer_name + '/w', weight)  # 统计w
        with tf.name_scope('b'):
            b = tf.Variable(tf.zeros([1, out_size]) + 0.1)
            tf.summary.histogram(layer_name + '/b', b)  # 统计b
        with tf.name_scope('w_plus_b'):
            weight_plus_b = tf.matmul(inputs, weight) + b
        if activation_function is None:
            output = weight_plus_b
        else:
            output = activation_function(weight_plus_b)
        tf.summary.histogram(layer_name + '/output', output)  # 统计输出
        return output


# 创建网络的测试
# 包含了对tensorboard的命名块测试，以及训练过程统计
def test_build_networks():
    # train data
    x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
    y_data = np.square(x_data) + np.random.normal(0, 0.05, x_data.shape)

    # input and output
    with tf.name_scope('input'):
        with tf.name_scope('x_in'):
            x = tf.placeholder(tf.float32, [None, 1])
        with tf.name_scope('y_in'):
            y = tf.placeholder(tf.float32, [None, 1])

    # build net
    l1 = add_layer(x, 1, 10, tf.nn.relu, layer_name='l1')
    prediction = add_layer(l1, 10, 1, layer_name='l2', activation_function=None)
    # loss function
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(prediction - y), axis=1))
        tf.summary.scalar('loss', loss)  # 这样写会将loss的统计图放在events下面
    # optimizer
    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

    # mainloop
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        merged = tf.summary.merge_all()  # 合并所有的训练图
        writer = tf.summary.FileWriter('./logs/', sess.graph)  # 输出graph
        for th_ in range(MAX_STEP):
            sess.run(optimizer, feed_dict={x: x_data, y: y_data})
            if th_ % 50 == 0:
                print('train_loss:', sess.run(loss, feed_dict={x: x_data, y: y_data}))
                rs = sess.run(merged, feed_dict={x: x_data, y: y_data})  # 需要记录训练的数据，才能画出比较完整的统计图
                writer.add_summary(rs, th_)

        # test
        x_test = np.linspace(0, 1, 30, dtype=np.float32)[:, np.newaxis]
        y_test = np.square(x_test)
        fig = plt.figure()

        plt.plot(x_test, y_test)
        predict_value = sess.run(prediction, feed_dict={x: x_test})
        plt.plot(x_test, predict_value, 'r-', lw=3)
        plt.show()
        # print('predict:\n', sess.run(prediction, feed_dict={x: x_test}))


# 用mnist测试神经网络。TF的mnist中包含了训练集、测试集、随机抽样等函数，用起来很方便
# 可以参考的代码 https://github.com/MorvanZhou/tutorials/blob/master/tensorflowTUT/tf16_classification/full_code.py
def test_MNIST_classification():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST-data', one_hot=True)
    xs = tf.placeholder(tf.float32, [None, 28 * 28])  # mnist的图像尺寸
    ys = tf.placeholder(tf.float32, [None, 10])  # mnist的分类结果
    # build net
    # 实验发现，激活函数对网络的训练结果有很大影响。如果选择tf.nn.relu，则正确率不超过10%
    prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax, layer_name='l1')
    # loss
    loss = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), axis=1))
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    # main loop
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for th_ in range(MAX_STEP):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(optimizer, feed_dict={xs: batch_xs, ys: batch_ys})
            if th_ % 50 == 0:
                # 利用mnist数据集中的测试集，测试训练的精确度
                x_test = mnist.test.images
                y_test = mnist.test.labels
                prediction_value = sess.run(prediction, feed_dict={xs: x_test})
                compute_accuracy = tf.equal(tf.arg_max(prediction_value, dimension=1), tf.arg_max(y_test, dimension=1))
                accuracy_value = tf.reduce_mean(tf.cast(compute_accuracy, tf.float32))
                print('accuracy:', sess.run(accuracy_value, feed_dict={xs: x_test, ys: y_test}))


if __name__ == "__main__":
    # linear_regression()
    # test_placeholder()
    # linear_official_test()
    # test_build_networks()
    test_MNIST_classification()
