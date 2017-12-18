import argparse
import sys
import os
import time

import numpy as np
import tensorflow as tf

import mnist

TRAIN_SET_NUM = 60000
ETA = 0.01
TRAIN_FILE = 'train_50_bit_flip'
TEST_FILE = 'test_5_bit_flip'

#files to run the fully connected neural network for error correction
#need to specify all the file paths first
def run_training():

    with tf.Graph().as_default():

        session = tf.InteractiveSession()

        train_fname_img = os.path.join("", TRAIN_FILE)
        train_img_set = []
        with open(train_fname_img, 'rb') as tfimg:
            tfimg.read(16)
            for i in range(0,TRAIN_SET_NUM):
                dataline = tfimg.read(784)
                intline = list(dataline)
                train_img_set.append(intline)

        target_fname_img = os.path.join("", 'train-images-idx3-ubyte')
        tar_img_set = []
        with open(target_fname_img, 'rb') as tarfimg:
            tarfimg.read(16)
            for i in range(0,TRAIN_SET_NUM):
                dataline = tarfimg.read(784)
                intline = list(dataline)
                tar_img_set.append(intline)

        test_fname_img = os.path.join("", TEST_FILE)
        test_img_set = []
        with open(test_fname_img, 'rb') as testfimg:
            testfimg.read(16)
            for i in range(0,10000):
                dataline = testfimg.read(784)
                intline = list(dataline)
                test_img_set.append(intline)

        #begin neural networks
        W1 = tf.Variable(tf.random_uniform([1568,784]))
        b1 = tf.Variable(tf.random_uniform([1568,1]))
        W2 = tf.Variable(tf.random_uniform([784,1568]))
        b2 = tf.Variable(tf.random_uniform([784,1]))

        W1n = W1
        b1n = b1
        W2n = W2
        b2n = b2

        init = tf.global_variables_initializer()
        session.run(init)

        #timer for time calculation
        start_time = time.time()
        end_time = time.time()

        for iter in range(0,TRAIN_SET_NUM):
            print("iteration ",iter)

            end_time = time.time()
            print("time ",end_time - start_time)
            start_time = time.time()

            x = [train_img_set[iter]]
            x = tf.transpose(x)
            x = tf.cast(x,tf.float32)

            b1 = b1n
            b2 = b2n
            W1 = W1n
            W2 = W2n

            u1 = tf.matmul(W1,x)
            u1 = tf.add(u1,b1)
            y1 = u1
            y1 = tf.nn.relu(y1)

            u2 = tf.matmul(W2,y1)
            u2 = tf.add(u2,b2)
            u2_trans = tf.transpose(u2)
            k2list = (u2_trans.eval())[0]
            k2min = min(k2list)
            y2 = tf.add(u2,-k2min)
            y2_trans = tf.transpose(y2)
            k2list2 = (y2_trans.eval())[0]
            k2max2 = max(k2list2)
            k2scale2 = 255/k2max2
            y2 = tf.scalar_mul(k2scale2,y2)
            y2 = tf.nn.relu(y2)

            session.run(y2)

            target = [tar_img_set[iter]]
            target = tf.transpose(target)
            target = tf.cast(target,tf.float32)

            err = tf.subtract(y2,target)

            y2primem = []
            trans_u2 = tf.transpose(u2)
            trans_u2eval = trans_u2.eval()
            for ele in trans_u2eval[0]:
                if ele > 0:
                    y2primem.append(k2scale2)
                else:
                    y2primem.append(0)

            y2primem = tf.transpose([y2primem])
            y2primem = tf.cast(y2primem,tf.float32)

            de_db2 = tf.multiply(y2primem,err)
            b2n = tf.subtract(b2,tf.scalar_mul(ETA,de_db2))

            de_dw2 = tf.matmul(de_db2,tf.transpose(y1))
            W2n = tf.subtract(W2,tf.scalar_mul(ETA,de_dw2))

            y1primem = []
            trans_u1 = tf.transpose(u1)
            for ele in trans_u1.eval()[0]:
                if ele > 0:
                    y1primem.append(1)
                else:
                    y1primem.append(0)
            y1primem = tf.transpose([y1primem])
            y1primem = tf.cast(y1primem,tf.float32)

            de_db1 = tf.multiply(tf.matmul(tf.transpose(W2),de_db2),y1primem)
            b1n = tf.subtract(b1,tf.scalar_mul(ETA,de_db1))

            de_dw1 = tf.matmul(de_db1,tf.transpose(x))
            W1n = tf.subtract(W1,tf.scalar_mul(ETA,de_dw1))

            session.run(W1n)

            if((iter % 100) ==0):
                print("testing")
                xt = [test_img_set[iter % 10000]]
                xt = tf.cast(xt,tf.float32)
                xt = tf.transpose(xt)

                u1t = tf.matmul(W1,xt)
                u1t = tf.add(u1t,b1)
                y1t = tf.nn.relu(u1t)

                u2t = tf.matmul(W2,y1t)
                u2t = tf.add(u2t,b2)
                u2t_trans = tf.transpose(u2t)
                k2tlist = (u2t_trans.eval())[0]
                k2tmin = min(k2tlist)
                y2t = tf.add(u2t,-k2tmin)
                y2t_trans = tf.transpose(y2t)
                k2tlist2 = (y2t_trans.eval())[0]
                k2tmax2 = max(k2tlist2)
                k2tscale2 = 255/k2tmax2
                y2t = tf.scalar_mul(k2tscale2,y2t)
                y2t = tf.nn.relu(y2t)

                imagem = y2t
                imagem = tf.transpose(imagem)
                image = imagem.eval()
                imagedata = image[0]
                mnist.show(np.reshape(imagedata,(28,28)))

def main(_):
    run_training()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)