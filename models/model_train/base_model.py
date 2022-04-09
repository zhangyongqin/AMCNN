
import cv2
from keras_preprocessing.image import array_to_img, np

from keras.models import *
from keras.layers.merge import concatenate, multiply, add
from keras.layers import Input,Conv2D,BatchNormalization, MaxPooling2D
from keras.layers import Conv2DTranspose,Dropout,LeakyReLU
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
import keras.backend as K
from data import *
from model.data import dataProcess



def dice_coef(y_true, y_pred ):
    y_pred[y_pred > 0.5] = np.float32(1)
    y_pred[y_pred < 0.5] = np.float32(0)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 0.00001) / (K.sum(y_true_f) + K.sum(y_pred_f) + 0.00001)


def dice_score(seg, gt, ratio=0.5):
    """
    function to calculate the dice score
    """
    seg = seg.flatten()
    gt = gt.flatten()
    seg[seg > ratio] = np.float32(1)
    seg[seg < ratio] = np.float32(0)
    gt[gt > ratio] = np.float(1)
    gt[gt < ratio] = np.float(0)
    dice = float(2 * (gt * seg).sum())/float(gt.sum() + seg.sum())
    return dice


def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same'):
    x = Conv2D(nb_filter, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x


def Conv2dT_BN(x, filters, kernel_size, strides=(2, 2), padding='same'):
    x = Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x


def Conv2d_XH(cishu,x, filters, kernel_size, strides=(2, 2), padding='same'):

    conv1 = Conv2d_BN(x,filters,(3,3))
    conv2 = Conv2d_BN(conv1,filters,(3,3))
    conv3 = Conv2d_BN(conv2,filters,(3,3))

    for i in range(0,cishu):
        zconv1 = Conv2d_BN(conv3, filters/2, (3, 3))
        conv1 = concatenate([conv1, zconv1], axis=3)
        conv1 = Conv2d_BN(conv1,filters,(1,1))
        zconv2 = Conv2d_BN(conv1, filters/2, (3, 3))
        conv2 = concatenate([conv2, zconv2], axis=3)
        conv2 = Conv2d_BN(conv2, filters, (1, 1))
        zconv3 = Conv2d_BN(conv2, filters/2, (3, 3))
        conv3 = concatenate([conv3, zconv3], axis=3)
        conv3 = Conv2d_BN(conv3, filters, (1, 1))

    return conv3


def attention_C(x,filter):
    shoutcut = x
    conv1 = Conv2d_BN(x,filter,(1,1))
    conv2 = Conv2d_BN(x,filter,(3,3))
    conv3 = Conv2d_BN(x,filter,(5,5))

    conv11 = Conv2d_BN(conv1,filter,(1,1))
    conv12 = Conv2d_BN(conv1,int(filter/2),(3,3))
    conv13 = Conv2d_BN(conv1,int(filter/3),(5,5))

    conv21 = Conv2d_BN(conv2,int(filter/2),(1,1))
    conv22 = Conv2d_BN(conv2,int(filter/3),(3,3))
    conv23 = Conv2d_BN(conv2,int(filter/2),(5,5))

    conv31 = Conv2d_BN(conv3,int(filter/3),(1,1))
    conv32 = Conv2d_BN(conv3,int(filter/2),(3,3))
    conv33 = Conv2d_BN(conv3,filter,(5,5))

    convt1 = conv11
    convt2 = concatenate([conv12,conv21],axis=3)
    convt3 = concatenate([conv13,conv22,conv31],axis=3)
    convt4 = concatenate([conv23,conv32])
    convt5 = conv33

    convtZ = concatenate([convt1,convt2,convt3,convt4,convt5],axis=3)
    att_map = Conv2D(filters=filter, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(convtZ)
    out1 = multiply([x, att_map])
    out = add([out1, shoutcut])
    out = LeakyReLU(alpha=0.1)(out)

    return out


class myUnet(object):
    def __init__(self, img_rows = 128, img_cols = 128 ):
        self.img_rows = img_rows
        self.img_cols = img_cols
# 参数初始化定义
    def load_data(self):
        mydata = dataProcess(self.img_rows, self.img_cols)
        imgs_train, imgs_mask_train = mydata.load_train_data()
        imgs_test = mydata.load_test_data()
        return imgs_train, imgs_mask_train, imgs_test

# 载入数据
    def get_unet(self):
        inputs = Input((self.img_rows, self.img_cols, 1))

        conv1 = Conv2d_BN(inputs, 64, (3, 3))
        conv1 = Conv2d_BN(conv1, 64, (3, 3))
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)

        conv2 = Conv2d_BN(pool1, 128, (3, 3))
        conv2 = Conv2d_BN(conv2, 128, (3, 3))
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)

        conv3 = Conv2d_BN(pool2, 256, (3, 3))
        conv3 = Conv2d_BN(conv3, 256, (3, 3))
        pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3)

        conv4 = Conv2d_BN(pool3, 512, (3, 3))
        conv4 = Conv2d_BN(conv4, 512, (3, 3))
        pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv4)

        conv5 = Conv2d_BN(pool4, 1024, (3, 3))
        conv5 = Dropout(0.5)(conv5)
        conv5 = Conv2d_BN(conv5, 1024, (3, 3))
        conv5 = Dropout(0.5)(conv5)

        convt1 = Conv2dT_BN(conv5, 512, (3, 3))
        concat1 = concatenate([conv4, convt1], axis=3)
        concat1 = Dropout(0.5)(concat1)
        conv6 = Conv2d_BN(concat1, 512, (3, 3))
        conv6 = Conv2d_BN(conv6, 512, (3, 3))

        convt2 = Conv2dT_BN(conv6, 128, (3, 3))
        concat2 = concatenate([conv3, convt2], axis=3)
        concat2 = Dropout(0.5)(concat2)
        conv7 = Conv2d_BN(concat2, 256, (3, 3))
        conv7 = Conv2d_BN(conv7, 256, (3, 3))

        convt3 = Conv2dT_BN(conv7, 128, (3, 3))
        concat3 = concatenate([conv2, convt3], axis=3)
        concat3 = Dropout(0.5)(concat3)
        conv8 = Conv2d_BN(concat3, 128, (3, 3))
        conv8 = Conv2d_BN(conv8, 128, (3, 3))

        convt4 = Conv2dT_BN(conv8, 64, (3, 3))
        concat4 = concatenate([conv1, convt4], axis=3)
        concat4 = Dropout(0.5)(concat4)
        conv9 = Conv2d_BN(concat4, 64, (3, 3))
        conv9 = Conv2d_BN(conv9, 64, (3, 3))
        conv9 = Dropout(0.5)(conv9)
        outpt = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(conv9)

        model = Model(input=inputs, output=outpt)
        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    # 如果需要修改输入的格式，那么可以从以下开始修改，上面的结构部分不需要修改
    def train(self):
        print("loading data")
        imgs_train, imgs_mask_train, imgs_test = self.load_data()
        print("loading data done")
        model = self.get_unet()
        print("got unet")
        model.load_weights('Munet.hdf5')
        model_checkpoint = ModelCheckpoint('Munet.hdf5', monitor='val_loss',verbose=1, save_best_only=True)
        print('Fitting model...')
        # model.fit(imgs_train, imgs_gtruth_train, batch_size=batch_size, nb_epoch=nb_epochs, verbose=1, validation_data=(imgs_val,imgs_gtruth_val), shuffle=True, callbacks=callbacks_list)
        # model.fit(imgs_train, imgs_mask_train, batch_size=2, epochs=30, verbose=1, validation_split=0.2, shuffle=True,
        #           callbacks=[model_checkpoint])
        print('predict test data')
        imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
        np.save('../results/imgs_mask_test.npy', imgs_mask_test)

    def save_img(self):
        print("array to train")
        imgs = np.load('../results/imgs_mask_test.npy')
        sum = 0.0
        for i in range(imgs.shape[0]):
            img = imgs[i]
            labels = cv2.imread('../Tlabel/%d.png' % (i),0)
            print(labels.shape)
            print(i)

            labels = labels.astype(np.float32) / 255
            dice = dice_score(labels, img)
            print(dice)
            sum = sum + dice
            img = array_to_img(img)
            if(i<100):
                img.save("../results/%d.png" % (i))

        avgdice = sum/(imgs.shape[0])
        print(imgs.shape[0])
        print('平均dice系数：')
        print(avgdice)


if __name__ == '__main__':
    myunet = myUnet()
    myunet.train()
    myunet.save_img()
