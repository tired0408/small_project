from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format("channels_first")
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from deepLearning.session_four.fr_utils import *
# np.set_printoptions(threshold=np.nan)

FRmodel = faceRecoModel(input_shape=(3, 96, 96))
print("Total Parames:", FRmodel.count_params())
def triplet_loss(y_ture, y_pred, alpha=0.2):

    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=1)

    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=1)

    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    #reduce_sum( ) 是求和函数，在 tensorflow 里面，计算的都是 tensor，可以通过调整 axis =0,1 的维度来控制求和维度
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0))

    return loss
# with tf.Session() as test:
#     tf.set_random_seed(1)
#     y_true = (None, None, None)
#     y_pred = (tf.random_normal([3, 128], mean=6, stddev=0.1, seed=1),
#               tf.random_normal([3, 128], mean=1, stddev=1, seed=1),
#               tf.random_normal([3, 128], mean=3, stddev=4, seed=1))
#     loss = triplet_loss(y_true, y_pred)
#     print("loss = " + str(loss.eval()))

FRmodel.compile(optimizer="adam", loss=triplet_loss, metrics=["accuracy"])
load_weights_from_FaceNet(FRmodel)
database = {}
database["danielle"] = img_to_encoding("images/danielle.png", FRmodel)
database["younes"] = img_to_encoding("images/younes.jpg", FRmodel)
database["tian"] = img_to_encoding("images/tian.jpg", FRmodel)
database["andrew"] = img_to_encoding("images/andrew.jpg", FRmodel)
database["kian"] = img_to_encoding("images/kian.jpg", FRmodel)
database["dan"] = img_to_encoding("images/dan.jpg", FRmodel)
database["sebastiano"] = img_to_encoding("images/sebastiano.jpg", FRmodel)
database["bertrand"] = img_to_encoding("images/bertrand.jpg", FRmodel)
database["kevin"] = img_to_encoding("images/kevin.jpg", FRmodel)
database["felix"] = img_to_encoding("images/felix.jpg", FRmodel)
database["benoit"] = img_to_encoding("images/benoit.jpg", FRmodel)
database["arnaud"] = img_to_encoding("images/arnaud.jpg", FRmodel)

def verify(image_path, identity, database, model):

    encoding = img_to_encoding(image_path, FRmodel)

    dist = np.linalg.norm(database[identity] - encoding)

    if dist < 0.7:
        print("It's " + str(identity) + ", welcome home!")
        door_open = True
    else:
        print("It's not " + str(identity) + ", please go away")
        door_open = False

    return dist, door_open

# verify("images/camera_0.jpg", "younes", database, FRmodel)
def who_is_it(image_path, database, model):

    encoding = img_to_encoding(image_path, FRmodel)

    min_dist = 100

    for (name, db_enc) in database.items():

        dist = np.linalg.norm(encoding - db_enc)

        if min_dist > dist:
            min_dist = dist
            identity = name

    if min_dist > 0.7:
        print("Not in the database")
    else:
        print("it's " + str(identity) + ", the distance is " + str(min_dist))

    return min_dist, identity
# who_is_it("images/camera_0.jpg", database, FRmodel)
