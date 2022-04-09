import tensorflow as tf
sess = tf.Session()
a = tf.constant(1)
b = tf.constant(2)
print(sess.run(a+b))
