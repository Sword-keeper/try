import tensorflow as tf
import numpy as  np
import matplotlib.pyplot as plt   
def add_layer(inputs,in_size,out_size,activation_function = None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size])) #生成一个in_s *out_s的矩阵
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)            #一行，out_s列 机器学习中bias推荐值不为0
    Wx_plus_b = tf.matmul(inputs,Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

x_data = np.linspace(-1,1,300)[:,np.newaxis]  #后面的是生成新维度，见note， -1到1均匀300份
noise = np.random.normal(0,0.05,x_data.shape)  # 以0为中心，方差 0.05，输出的shape和x_data一样的值

y_data = np.square(x_data) - 0.5 +noise



xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])
#输入层一个神经元（因为只有x_data这一个属性），隐藏层10个神经元，输出层一个（因为也只有ydata这一个属性）
l1 = add_layer(xs,1,10,activation_function = tf.nn.relu)
#隐藏层  输入xdata ins为1，输出十个（之前定义隐藏层有10个神经元）。 
prediction = add_layer(l1,10,1,activation_function = None)
#输出层  输入的是隐藏层计算出的数据， 返回的outputs 十个

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices = [1]))
#reduce_sum就是求和 是tf的函数， mean就是求平均值。reduction_indices 见函数 【1】是压缩成一行
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#训练，梯度下降方法 学习效率0.1 是让loss最小

init = tf.global_variables_initializer()
sess = tf.Session()
#with tf.Session() as sess:
sess.run(init)


fig= plt.figure()            #先生成一个图片框
ax = fig.add_subplot(1,1,1)   #连续性的画图，分成一行一列 在第一快图中
ax.scatter(x_data,y_data)      #以点的形式在图中打印出来
plt.ion()
plt.show()   #没有上一句的话，只是静态的图像，加上ion后，会不断变化。

for i in range(1000):
    sess.run(train_step,feed_dict = {xs:x_data,ys:y_data})
    if i%50==0:
       # print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
            #只要是通过ph进行运算的 都要feed
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction,feed_dict={xs:x_data}) #这是预测的值
        lines = ax.plot(x_data,prediction_value,'r-',lw=5)       #红线，宽度为5
        #ax.lines.remove(lines[0])
        plt.pause(0.1)
        
        










        
