import os

from random import randint

FILE_NAME = 'train-images-idx3-ubyte'
INTERATION = 60000
RANDOM_BITS = 50

#functions for write the error data file
#need to specify the FILE_NAME path first
def main():
    fname_img = os.path.join("", FILE_NAME)
    with open(fname_img, 'rb') as fimg:
        with open("temp", 'wb') as wimg:
            wimg.write(fimg.read(16))
            for i in range(0,INTERATION):
                dataline = fimg.read(784)
                dataarray = list(dataline)
                for j in range(0,RANDOM_BITS):
                    ran = randint(0, 783)
                    dataarray[ran] = dataarray[ran] - 127
                    if dataarray[ran] < 0:
                        dataarray[ran] = dataarray[ran] + 255
                data = bytes(dataarray)
                wimg.write(data)
