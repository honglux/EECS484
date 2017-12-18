import argparse
import os
import sys
import struct

import tensorflow as tf
import numpy as np

import mnist

FLAGS=None
TRAIN_SET_NUM = 60000
TEST_SET_NUM = 10000
BATCH_SIZE = 100
TEST_FILE_NAME = "test_50_bit_flip"


#file for run memory correction
#need to specify all the file paths first
def run_training():

    lable_holder = tf.placeholder(tf.int32,shape=(TRAIN_SET_NUM,1))

    fname_img = os.path.join("", 'train-images-idx3-ubyte')
    fname_lbl = os.path.join("", 'train-labels-idx1-ubyte')
    lab_set = []
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        for i in range(0,TRAIN_SET_NUM):
            lbl = list(flbl.read(1))
            lab_set.append(lbl)

    img_set = []
    with open(fname_img, 'rb') as fimg:
        fimg.read(16)
        for i in range(0,TRAIN_SET_NUM):
            dataline = fimg.read(784)
            intline = list(dataline)
            img_set.append(intline)

    #classify and extract image;
    img_dict = {}
    for class_num in range(0,10):
        img_dict[class_num] = []
    for set_num in range(0,TRAIN_SET_NUM):
        for class_num in range(0,10):
            if(lab_set[set_num][0] == class_num):
                img_dict[class_num].append(img_set[set_num])

    mem_dict = {}
    mem_sum_dict = {}
    for class_num in range(0,10):
        mem_dict[class_num] = []
        mem_sum_dict[class_num] = tf.Variable(tf.zeros([784,784]))
        mem_sum_dict[class_num] = tf.cast(mem_sum_dict[class_num],tf.float32)


    test_file_path = os.path.join("", TEST_FILE_NAME)
    test_lable_path = os.path.join("", 't10k-labels-idx1-ubyte')
    test_lab_set = []
    with open(test_lable_path, 'rb') as test_flbl:
        magic, num = struct.unpack(">II", test_flbl.read(8))
        for i in range(0,TEST_SET_NUM):
            lbl = list(test_flbl.read(1))
            test_lab_set.append(lbl)

    test_img_set = []
    with open(test_file_path, 'rb') as test_fimg:
        test_fimg.read(16)
        for i in range(0,TEST_SET_NUM):
            dataline = test_fimg.read(784)
            intline = list(dataline)
            test_img_set.append(intline)

    #classify and extract test image;
    test_img_dict = {}
    for class_num in range(0,10):
        test_img_dict[class_num] = []
    for set_num in range(0,TEST_SET_NUM):
        for class_num in range(0,10):
            if(test_lab_set[set_num][0] == class_num):
                test_img_dict[class_num].append(test_img_set[set_num])

    init = tf.global_variables_initializer()
    session = tf.InteractiveSession()
    session.run(init)


    iter = 0
    #average total 6000 numbers
    while iter < 5000:
        for class_num in range(0,10):
            image_holder = img_dict[class_num][iter:iter+BATCH_SIZE]
            for image in image_holder:
                x = image
                xn = tf.add(x,-127)
                xn = tf.cast(xn,tf.float32)
                xns = tf.scalar_mul(float(1/127),xn)
                xns = [xns]
                mem_mat = tf.matmul(xns,xns,transpose_a=True)
                mem_dict[class_num].append(mem_mat)
                mem_sum_dict[class_num] = tf.add(mem_sum_dict[class_num],mem_mat)
        session.run(mem_sum_dict)

        for class_num in range(0,10):
            test_img = test_img_dict[class_num][iter]
            test_img = tf.cast(test_img,tf.float32)
            test_img = tf.add(test_img,-127)
            test_img = tf.scalar_mul(float(1/127),test_img)
            test_img = [test_img]
            result = tf.matmul(mem_sum_dict[class_num],test_img,transpose_b=True)
            result = tf.transpose(result)
            scalar = list(result[0].eval())[0]
            result = tf.scalar_mul(float(1/(scalar)),result)
            result = tf.scalar_mul(float(127),result)
            result = tf.scalar_mul(-1,result)
            result = tf.add(result,128)
            result = tf.cast(result,tf.int32)
            session.run(result)
            mnist.show(np.reshape(list(result[0].eval()),(28,28)))

        iter += BATCH_SIZE


def main(_):
    run_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)