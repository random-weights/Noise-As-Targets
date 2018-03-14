import numpy as np
import tensorflow as tf
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
import matplotlib.pyplot as plt
import itertools

def generate_Y(count,dim):
    """
    Generates y vector of size (count,dim)
    each entry of y is a point on unit circle.
    How??
    generate a random int vector and divide by its eucledian distance.
    :param count: the no.of images usually dictated by one_hot_length
    :param dim: the dimension of each y vector (usually in power of 2)
    :return: a vector Y that is the target mass
    """
    Y = np.random.randint(1,100,size = [count,dim]).astype(np.float32)
    origin = np.zeros(shape = [1,dim])
    for y_numb, y_vector in enumerate(Y):
        dist = distance.euclidean(y_vector, origin)
        Y[y_numb] = y_vector/dist
    return Y

def Hungerian(y,z):
    """
    Assert y and z have same shape. The cost matrix will be computed in this function.
    How to compute cost matrix:
        Use 2 for loops and calculate l2 distance from each z to each y.
    resulting cost matrix should be a square with dims (batch_size,batch_size) when y,z have dimensions (batch_size,d)
    :param y: target noise on unit sphere
    :param z: noise estimated from images(output of neural network)
    :return: y reordered such that z's will have corresponding y's
    """
    len_y = len(y)
    len_z = len(z)
    shape_y = y.shape
    if not len_y == len_z:
        raise AssertionError
    else:
        cost_matrix = np.zeros(shape = [len_y,len_y])
        for y_index, y_vect in enumerate(y):
            for z_index, z_vect in enumerate(z):
                cost_matrix[z_index][y_index] = distance.euclidean(y_vect,z_vect)
        res = linear_sum_assignment(cost_matrix)
        new_indices = res[1]
        new_y = []
        for index in new_indices:
            new_y.append(y[index])
        return np.array(new_y).reshape(shape_y)

one_hot_length = 60000
class Data():
    def __init__(self,train_fname):
        self.X_TRAIN_FNAME = train_fname

    def index_to_oh(self,index):
        """
        :param index: the index of the image in csv file
        :return: a one hot vector that indicates position of image in csv fie
        """
        vector = [0]*one_hot_length
        vector[index] = 1
        vector = np.array(vector).reshape(1,one_hot_length)
        return vector

    def image(self):
        """
        get the next image in csv file.
        :return: an image as numpy array with dims (1,28,28,1)
        """
        fh = open(self.X_TRAIN_FNAME,'r')
        for line in fh:
            line = line.strip("\n")
            pixels = line.split(",")
            norm_pixels = [int(pixel)/255.0 for pixel in pixels]
            norm_pixels = np.array(norm_pixels).astype(np.float32).reshape(1,28,28,1)
            yield norm_pixels

    def onehot(self):
        """
        get the next onehot array
        :return: onehot array od dims [1,one_hot_length]
        """
        indices = range(one_hot_length)
        for index in indices:
            yield self.index_to_oh(index)

    def image_batch(self,batch_size):
        """
        Uses the image generator.
        :param batch_size: recommended in power 2
        :return: a numpy array that contains <batch_size> images
        """
        image_gen = itertools.cycle(self.image())
        while True:
            image_arr = []
            for _ in range(batch_size):
                image_arr.append(next(image_gen))
            yield np.array(image_arr).reshape(batch_size,28,28,1)

    def onehot_batch(self,batch_size):
        """
        Uses onehot() generator.
        :param batch_size: recommended in power of 2
        :return: a numpy array of <batch_size> one_hot arrays.
        """
        oh_array = itertools.cycle(self.onehot())
        while True:
            onehot_batch = []
            for _ in range(batch_size):
                onehot_batch.append(next(oh_array))
            yield np.array(onehot_batch).reshape(batch_size,one_hot_length)

"""
Hyper parameters:
y_dim : the dimension of one y vector.
batch_size:
"""
y_dim = 128
batch_size = 64

gph = tf.Graph()
with gph.as_default():

    init = tf.group(tf.local_variables_initializer(),tf.global_variables_initializer(),name = "var_initializer")

    with tf.variable_scope("inputs"):
        y = tf.placeholder(tf.float32,shape = [batch_size,128],name = "y")
        img = tf.placeholder(tf.float32,shape= [batch_size,28,28,1],name = "input_img")

    kern_init = tf.random_normal_initializer(mean = 0.0, stddev=0.1)

    with tf.variable_scope("layer1"):
        kern1 = tf.get_variable("kern1",shape = [7,7,1,2],initializer=kern_init)
        l1 = tf.nn.conv2d(input = img,filter=kern1,strides = [1,1,1,1],padding="VALID",name = "conv1")
        l1_relu = tf.nn.relu(l1,"l1_relu")

    with tf.variable_scope("layer2"):
        kern2 = tf.get_variable("kern2",shape = [5,5,2,4],initializer=kern_init)
        l2 = tf.nn.conv2d(input = l1_relu,filter = kern2,strides = [1,1,1,1],padding="VALID")
        l2_relu = tf.nn.relu(l2,"l2_relu")

    with tf.variable_scope("layer3"):
        kern3 = tf.get_variable("kern3",shape = [5,5,4,8],initializer=kern_init)
        l3 = tf.nn.conv2d(input = l2_relu,filter = kern3,strides = [1,1,1,1],padding="VALID")
        l3_relu = tf.nn.relu(l3,"l3_relu")

    with tf.variable_scope("layer4"):
        kern4 = tf.get_variable("kern4",shape = [5,5,8,4],initializer=kern_init)
        l4 = tf.nn.conv2d(input = l3_relu,filter = kern4,strides = [1,1,1,1],padding="VALID")
        l4_relu = tf.nn.relu(l4,"l4_relu")

    with tf.variable_scope("layer5"):
        kern5 = tf.get_variable("kern5",shape = [3,3,4,2],initializer=kern_init)
        l5 = tf.nn.conv2d(input = l4_relu,filter = kern5,strides = [1,1,1,1],padding="VALID")
        l5_relu = tf.nn.relu(l5,"l5_relu")

    z = tf.contrib.layers.flatten(inputs = l5_relu,scope = "flat_tensor")

    # unit normalize each vector in z
    with tf.variable_scope("normalize"):
        dist = tf.sqrt(tf.reduce_sum(z ** 2, axis=[1]), name="euclidian_dist")
        dist = tf.reshape(dist, shape=[batch_size, 1],name = "euclidian_dist_reshape")
        z_norm = tf.div(z, dist,name = "z_norm")

    # cost is a l2 loss
    cost = tf.reduce_mean((z_norm - y)**2,name = "cost")
    tf.summary.scalar("cost", cost)

    train = tf.train.AdamOptimizer(1e-2).minimize(cost,name = "optimizer")

    saver = tf.train.Saver()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("checkpoint", graph=gph)

with tf.Session(graph = gph) as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    Y_norm = generate_Y(one_hot_length,128)
    train_data = Data(r"D:\data\mnist\x_train.csv")

    image_batch_gen = train_data.image_batch(batch_size)
    onehot_batch_gen = train_data.onehot_batch(batch_size)
    for epoch in range(10000):
        print("\rEpoch: ".format(epoch)+str(epoch),end = "")
        onehot_array = next(onehot_batch_gen)
        image_array = next(image_batch_gen)

        z_vector = sess.run(z_norm,feed_dict = {img: image_array})
        y_vector = np.matmul(onehot_array,Y_norm)
        y_vector = Hungerian(y_vector,z_vector)

        feed_dict = {img: image_array,
                     y: y_vector}
        _,summary = sess.run([train,merged],feed_dict=feed_dict)

        if (epoch+1)%100 == 0:
            writer.add_summary(summary,epoch)

    saver.save(sess,"checkpoint/NAT")
    print("Training done")