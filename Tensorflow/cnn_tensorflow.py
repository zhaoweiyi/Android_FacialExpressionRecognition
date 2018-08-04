from util import getData, y2indicator
from sklearn.utils import shuffle
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util

#Weight Initialization

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
  
#Convolution and Pooling
  
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
  
def main():
  Xdata, Ydata = getData()
  
  K = len(set(Ydata)) # won't work later b/c we turn it into indicator
  print('k:', K)

  # make a validation set
  Xdata, Ydata = shuffle(Xdata, Ydata)
  Xdata = Xdata.astype(np.float32)
  Ydata = y2indicator(Ydata).astype(np.float32)

  Xvalid, Yvalid = Xdata[-1000:], Ydata[-1000:]
  Xtrain, Ytrain = Xdata[:-1000], Ydata[:-1000]

  # initialize hidden layers
  N, D = Xtrain.shape
  print('N:', N)
  print('D:', D)
  
  x = tf.placeholder(tf.float32, shape=[None, D], name="input_x")
  y_ = tf.placeholder(tf.float32, shape=[None, K], name="input_y")
  
  #First Convolutional Layer
  
  W_conv1 = weight_variable([5, 5, 1, 32])
  b_conv1 = bias_variable([32])
  x_image = tf.reshape(x, [-1, 48, 48, 1])

  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  
  h_pool1 = max_pool_2x2(h_conv1)
  
  keep_conv_prob = tf.placeholder_with_default(0.9, shape=(), name="keep_conv_prob")
  h_pool1_drop = tf.nn.dropout(h_pool1, keep_conv_prob)
  
  #Second Convolutional Layer

  W_conv2 = weight_variable([3, 3, 32, 64])
  b_conv2 = bias_variable([64])

  h_conv2 = tf.nn.relu(conv2d(h_pool1_drop, W_conv2) + b_conv2)
  h_pool2 = max_pool_2x2(h_conv2)
  
  h_pool2_drop = tf.nn.dropout(h_pool2, keep_conv_prob)
  
  #Third Convolutional Layer
  
  W_conv3 = weight_variable([3, 3, 64, 64])
  b_conv3 = bias_variable([64])

  h_conv3 = tf.nn.relu(conv2d(h_pool2_drop, W_conv3) + b_conv3)
  h_pool3 = max_pool_2x2(h_conv3)
  h_pool3_flat = tf.reshape(h_pool3, [-1, 6*6*64])
  
  keep_prob = tf.placeholder_with_default(0.5, shape=(), name="keep_prob")
  h_pool3_flat_drop = tf.nn.dropout(h_pool3_flat, keep_prob)
  
  #First Densely Connected Layer

  W_fc1 = weight_variable([6 * 6 * 64, 1024])
  b_fc1 = bias_variable([1024])
  
  h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat_drop, W_fc1) + b_fc1)

  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
  
  #Second Densely Connected Layer

  W_fc2 = weight_variable([1024, 512])
  b_fc2 = bias_variable([512])
  
  h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
  
  h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
  
  #Output Layer
  
  W_fc3 = weight_variable([512, K])
  b_fc3 = bias_variable([K])

  y_conv = tf.add(tf.matmul(h_fc2_drop, W_fc3), b_fc3, name="outlayer")
  prob = tf.nn.softmax(y_conv, name="probability")
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
  train_step = tf.train.AdamOptimizer(10e-4).minimize(cross_entropy)
  pred = tf.argmax(prob, 1, output_type="int32", name="predict")  # 输出结点名称predict方便后面保存为pb文件
  correct_prediction = tf.equal(pred, tf.argmax(y_, 1, output_type='int32'))
  #pred = tf.argmax(y_conv, 1, name="predict")
  #correct_prediction = tf.equal(pred, tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  
  batch_sz = 200
  
  n_batches = N // batch_sz

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(150):# 200 epochs
      Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
      for j in range(n_batches):
        Xbatch = Xtrain[j*batch_sz:(j*batch_sz+batch_sz)]
        Ybatch = Ytrain[j*batch_sz:(j*batch_sz+batch_sz)]
        if j % 50 == 0:
          train_cost = cross_entropy.eval(feed_dict={x: Xbatch, y_: Ybatch, keep_conv_prob: 1.0, keep_prob: 1.0})         
          train_accuracy = accuracy.eval(feed_dict={x: Xbatch, y_: Ybatch, keep_conv_prob: 1.0, keep_prob: 1.0})
          test_accuracy = accuracy.eval(feed_dict={x: Xvalid, y_: Yvalid, keep_conv_prob: 1.0, keep_prob: 1.0})
          test_cost = cross_entropy.eval(feed_dict={x: Xvalid, y_: Yvalid, keep_conv_prob: 1.0, keep_prob: 1.0})
          print('epoch %d, training cost %g, test cost %g, training accuracy %g, test accuracy %g' % (i, train_cost, test_cost, train_accuracy, test_accuracy))
        train_step.run(feed_dict={x: Xbatch, y_: Ybatch, keep_conv_prob: 0.8, keep_prob: 0.5})

    #save model
    saver = tf.train.Saver({'W_conv1':W_conv1, 'b_conv1':b_conv1, 'W_conv2':W_conv2, 'b_conv2':b_conv2, 'W_conv3':W_conv3, 'b_conv3':b_conv3, 'W_fc1':W_fc1, 'b_fc1':b_fc1, 'W_fc2':W_fc2, 'b_fc2':b_fc2, 'W_fc3':W_fc3, 'b_fc3':b_fc3})
    saver.save(sess, "model_fer//fer_model.ckpt")

    #将训练好的模型保存为.pb文件，方便在Android studio中使用
    output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=['predict'])
    with tf.gfile.FastGFile('model_fer//FacialExpressionReg.pb', mode='wb') as f:  # ’wb’中w代表写文件，b代表将数据以二进制方式写入文件。
        f.write(output_graph_def.SerializeToString())

if __name__ == '__main__':
  main()
