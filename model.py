import tensorflow as tf

from tensorflow.keras import Input, Model
from tensorflow.keras.activations import sigmoid, tanh
from tensorflow.keras.layers import Conv2D, ReLU, Concatenate
from utils import *

class RFDNNet(Model):
    def __init__(self):
        super(RFDNNet, self).__init__()

    def SRB(self, X, filter):
        X1 = ReLU()(Conv2D(filter, kernel_size=(3,3), padding='same')(X))
        return X + X1

    def RFDB(self, X):
        filter_left = int(list(X.shape)[-1]/2)
        filter_right = int(list(X.shape)[-1])
        left_1 = Conv2D(filter_left, kernel_size=(1,1))(X)
        right_1 = self.SRB(X, filter_right)

        left_2 = Conv2D(filter_left, kernel_size=(1,1))(right_1)
        right_2 = self.SRB(right_1, filter_right)

        left_3 = Conv2D(filter_left, kernel_size=(1,1))(right_2)
        right_3 = self.SRB(right_2, filter_right)

        right_final = Conv2D(filter_left, kernel_size=(3,3), padding='same')(right_3)

        concat = Concatenate(axis=-1)([left_1, left_2, left_3, right_final])

        concate_1 = Conv2D(filter_right, kernel_size=(1,1))(concat)

        return concate_1 + X

    def main_model(self, X, scale_factor):
        X1 = Conv2D(self.feat, kernel_size=(3,3), padding='same')(X)
        
        out_B1 = self.RFDB(X1)
        out_B2 = self.RFDB(out_B1)
        out_B3 = self.RFDB(out_B2)
        out_B4 = self.RFDB(out_B3)

        concat = Concatenate(axis=-1)([out_B1,out_B3,out_B3,out_B4])

        concat_1 = Conv2D(self.feat, kernel_size=(1,1), activation='relu')(concat)

        LR = Conv2D(self.feat, kernel_size=(3,3), padding='same')(concat_1) + X1

        X_up = Conv2D(self.filter * (scale_factor ** 2), 3, padding='same')(LR)
        out = tf.nn.depth_to_space(X_up, scale_factor)
        out = Conv2D(3, kernel_size=(1,1))(out)
        return out



