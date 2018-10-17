import tensorflow as tf
import numpy as  np
def add_layer(inputs,in_size,out_size,activation_function = None):
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size,out_size]),name='W') #生成一个in_s *out_s的矩阵
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1,out_size])+0.1)            #一行，out_s列 机器学习中bias推荐值不为0
        with tf.name_scope('biases'):
            Wx_plus_b = tf.matmul(inputs,Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
    return outputs


x_data = np.linspace(-1,1,300)[:,np.newaxis]  #后面的是生成新维度，见note， -1到1均匀300份
y_data = np.square(x_data) - 0.5
noise = np.random.normal(0,0.05,x_data.shape)  # 以0为中心，方差 0.05，输出的shape和x_data一样的值
with tf.name_scope('input'):
    xs = tf.placeholder(tf.float32,[None,1],name ='x_input')
    ys = tf.placeholder(tf.float32,[None,1],name = 'y_input')
#输入层一个神经元（因为只有x_data这一个属性），隐藏层10个神经元，输出层一个（因为也只有ydata这一个属性）
l1 = add_layer(xs,1,10,activation_function = tf.nn.relu)
#隐藏层  输入xdata ins为1，输出十个（之前定义隐藏层有10个神经元）。 
prediction = add_layer(l1,10,1,activation_function = None)
#输出层  输入的是隐藏层计算出的数据， 返回的outputs 十个
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices = [1]))
#reduce_sum就是求和 是tf的函数， mean就是求平均值。reduction_indices 见函数 【1】是压缩成一行
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#训练，梯度下降方法 学习效率0.1 是让loss最小

init = tf.global_variables_initializer()
sess = tf.Session()
writer = tf.summary("logs/",sess.graph)

#with tf.Session() as sess:
#sess.run(init)

'''for i in range(1000):
    sess.run(train_step,feed_dict = {xs:x_data,ys:y_data})
    if i%50:
        print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
            #只要是通过ph进行运算的 都要feed'''
