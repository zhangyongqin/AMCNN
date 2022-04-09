import cv2
from keras_preprocessing.image import  np
from tensorflow.keras.models import *
from tensorflow.keras.layers import concatenate, multiply, add
from tensorflow.keras.layers import Conv2D,BatchNormalization, MaxPooling2D,AvgPool2D,MaxPool2D
from tensorflow.keras.layers import Conv2DTranspose,Dropout,LeakyReLU
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
from tensorflow.python.keras import Input
from tensorflow.python.keras.preprocessing.image import array_to_img
from keras_embed_sim import EmbeddingRet, EmbeddingSim

from model.data import dataProcess


def dice_coef(y_true, y_pred):
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
    dice = float(2 * (gt * seg).sum()) / float(gt.sum() + seg.sum())
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


def KongDong_BN(x, nb_filter, kernel_size, dilation, strides=(1, 1), padding='same'):
    x = Conv2D(nb_filter, kernel_size, strides=strides, padding=padding, dilation_rate=dilation)(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x


def kdInception(x, nb_filter, strides=(1, 1), padding='same'):
    conv1 = Conv2d_BN(x, nb_filter, (3, 3))
    KDconv1 = KongDong_BN(x, nb_filter, (3, 3), dilation=7)
    KDconv2 = KongDong_BN(x, nb_filter, (3, 3), dilation=9)

    x = concatenate([conv1, KDconv1, KDconv2], axis=3)
    x = Conv2d_BN(x, nb_filter, (1, 1))
    return x





def attention_C(x, filter):
    shoutcut = x
    conv1 = Conv2d_BN(x, filter, (1, 1))
    conv2 = Conv2d_BN(x, filter, (3, 3))
    conv3 = Conv2d_BN(x, filter, (5, 5))

    conv11 = Conv2d_BN(conv1, filter, (1, 1))
    conv12 = Conv2d_BN(conv1, int(filter / 2), (3, 3))
    conv13 = Conv2d_BN(conv1, int(filter / 3), (5, 5))

    conv21 = Conv2d_BN(conv2, int(filter / 2), (1, 1))
    conv22 = Conv2d_BN(conv2, int(filter / 3), (3, 3))
    conv23 = Conv2d_BN(conv2, int(filter / 2), (5, 5))

    conv31 = Conv2d_BN(conv3, int(filter / 3), (1, 1))
    conv32 = Conv2d_BN(conv3, int(filter / 2), (3, 3))
    conv33 = Conv2d_BN(conv3, filter, (5, 5))

    convt1 = conv11
    convt2 = concatenate([conv12, conv21], axis=3)
    convt3 = concatenate([conv13, conv22, conv31], axis=3)
    convt4 = concatenate([conv23, conv32])
    convt5 = conv33

    convtZ = concatenate([convt1, convt2, convt3, convt4, convt5], axis=3)
    att_map = Conv2D(filters=filter, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(convtZ)
    out1 = multiply([x, att_map])
    out = add([out1, shoutcut])
    out = LeakyReLU(alpha=0.1)(out)

    return out


def spatial_attention(channel_refined_feature):
    maxpool_spatial = KL.Lambda(lambda x: K.max(x, axis=3, keepdims=True))(channel_refined_feature)
    avgpool_spatial = KL.Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(channel_refined_feature)
    max_avg_pool_spatial = KL.Concatenate(axis=3)([maxpool_spatial, avgpool_spatial])
    spatial_attention_feature = KL.Conv2D(filters=1, kernel_size=(7, 7), padding="same", activation='sigmoid',
                                          kernel_initializer='he_normal', use_bias=False)(max_avg_pool_spatial)
    refined_feature = KL.Multiply()([channel_refined_feature, spatial_attention_feature])
    return KL.Add()([refined_feature, channel_refined_feature])


def attention_model_C(x, filter):
    shoutcut = x
    conv1 = Conv2d_BN(shoutcut, filter, (3, 3))
    conv21 = Conv2d_BN(conv1, filter, (1, 1))
    conv22 = Conv2d_BN(conv1, filter, (3, 3))
    conv31 = Conv2d_BN(conv22, filter, (1, 1))
    conv32 = Conv2d_BN(conv22, filter, (3, 3))
    conv41 = Conv2d_BN(conv32, filter, (1, 1))
    conv42 = Conv2d_BN(conv32, filter, (3, 3))
    #     conv21 = spatial_attention(conv21)
    #     conv31 = spatial_attention(conv31)
    #     conv41 = spatial_attention(conv41)
    #     conv42 = spatial_attention(conv42)
    convtz1 = concatenate([conv21, conv31, conv41, conv42], axis=3)
    convtz1 = Conv2d_BN(convtz1, filter, (1, 1))

    conv5 = Conv2d_BN(convtz1, filter, (3, 3))
    conv61 = Conv2d_BN(conv5, filter, (1, 1))
    conv62 = Conv2d_BN(conv5, filter, (3, 3))
    conv71 = Conv2d_BN(conv62, filter, (1, 1))
    conv72 = Conv2d_BN(conv62, filter, (3, 3))
    conv81 = Conv2d_BN(conv72, filter, (1, 1))
    conv82 = Conv2d_BN(conv72, filter, (3, 3))
    conv61 = spatial_attention(conv61)
    conv71 = spatial_attention(conv71)
    conv81 = spatial_attention(conv81)
    conv82 = spatial_attention(conv82)
    convtZ = concatenate([conv61, conv71, conv81, conv82], axis=3)
    convtZ = Conv2d_BN(convtZ, filter, (1, 1))

    att_map = Conv2D(filters=filter, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(convtZ)
    out1 = multiply([x, att_map])
    out = add([out1, shoutcut])
    out = LeakyReLU(alpha=0.1)(out)

    return out


def DenseAttention(x, filiter):
    conv1 = Conv2d_BN(x, filiter, (3, 3))
    concat1 = concatenate([x, conv1], axis=3)
    conv21 = Conv2d_BN(concat1, filiter, (1, 1))
    conv22 = Conv2d_BN(conv21, filiter, (3, 3))
    concat2 = concatenate([x, concat1, conv22], axis=3)
    conv3 = Conv2d_BN(concat2, filiter, (1, 1))

    conv4 = KongDong_BN(x, filiter, (3, 3), dilation=5)
    conv5 = KongDong_BN(x, filiter, (3, 3), dilation=7)
    concat3 = concatenate([conv3, conv4, conv5], axis=3)
    conv31 = Conv2d_BN(concat3, filiter, (1, 1))

    att_map = Conv2D(filters=filiter, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(conv31)
    out1 = multiply([x, att_map])
    out = add([out1, x])
    out = LeakyReLU(alpha=0.1)(out)

    return out


def attention_model(x, filter):
    shoutcut = x
    conv1 = Conv2d_BN(x, filter, (3, 3))
    conv2 = Conv2d_BN(x, filter, (5, 5))
    conv3 = Conv2d_BN(x, filter, (7, 7))
    conv4 = Conv2d_BN(x, filter, (9, 9))
    convtz1 = concatenate([conv1, conv2, conv3, conv4], axis=3)
    convtz1 = Conv2d_BN(convtz1, filter, (1, 1))
    conv5 = Conv2d_BN(convtz1, filter, (3, 3))
    conv6 = Conv2d_BN(convtz1, filter, (5, 5))
    conv7 = Conv2d_BN(convtz1, filter, (7, 7))
    conv8 = Conv2d_BN(convtz1, filter, (9, 9))

    convtZ = concatenate([conv5, conv6, conv7, conv8], axis=3)
    convtZ = Conv2d_BN(convtZ, filter, (1, 1))

    att_map = Conv2D(filters=filter, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(convtZ)
    out1 = multiply([x, att_map])
    out = add([out1, shoutcut])
    out = LeakyReLU(alpha=0.1)(out)

    return out


def attention_model_B(x, filter):
    shoutcut = x
    conv1 = Conv2d_BN(shoutcut, filter, (3, 3))
    conv21 = Conv2d_BN(conv1, filter, (3, 3))
    conv22 = Conv2d_BN(conv1, filter, (3, 3))
    conv31 = Conv2d_BN(conv22, filter, (3, 3))
    conv32 = Conv2d_BN(conv22, filter, (3, 3))
    conv41 = Conv2d_BN(conv32, filter, (3, 3))

    convtz1 = concatenate([conv1, conv21, conv31, conv41], axis=3)
    convtz1 = Conv2d_BN(convtz1, filter, (1, 1))

    conv5 = Conv2d_BN(convtz1, filter, (3, 3))
    conv61 = Conv2d_BN(conv5, filter, (3, 3))
    conv62 = Conv2d_BN(conv5, filter, (3, 3))
    conv71 = Conv2d_BN(conv62, filter, (3, 3))
    conv72 = Conv2d_BN(conv62, filter, (3, 3))
    conv81 = Conv2d_BN(conv72, filter, (3, 3))

    convtZ = concatenate([conv5, conv61, conv71, conv81], axis=3)
    convtZ = Conv2d_BN(convtZ, filter, (1, 1))

    att_map = Conv2D(filters=filter, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(convtZ)
    out1 = multiply([x, att_map])
    out = add([out1, shoutcut])
    out = LeakyReLU(alpha=0.1)(out)

    return out


class myUnet(object):
    def __init__(self, img_rows=128, img_cols=128):
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
        conv1 = kdInception(conv1, 64)
        conv1 = Conv2d_BN(conv1, 64, (3, 3))
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)
        conv1 = spatial_attention(conv1)


        conv2 = Conv2d_BN(pool1, 128, (3, 3))
        conv2 = kdInception(conv2, 128)
        conv2 = Conv2d_BN(conv2, 128, (3, 3))
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
        conv2 = spatial_attention(conv2)


        conv3 = Conv2d_BN(pool2, 256, (3, 3))
        conv3 = kdInception(conv3, 256)
        conv3 = Conv2d_BN(conv3, 256, (3, 3))
        pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3)
        conv3 = spatial_attention(conv3)


        conv4 = Conv2d_BN(pool3, 512, (3, 3))
        conv4 = kdInception(conv4, 512)
        conv4 = Conv2d_BN(conv4, 512, (3, 3))
        pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv4)
        conv4 = spatial_attention(conv4)


        conv5 = Conv2d_BN(pool4, 1024, (3, 3))
        conv5 = Dropout(0.5)(conv5)
        conv5 = Conv2d_BN(conv5, 1024, (3, 3))
        conv5 = Dropout(0.5)(conv5)

        convt1 = Conv2dT_BN(conv5, 512, (3, 3))
        concat1 = concatenate([conv4, convt1], axis=3)
        concat1 = Dropout(0.5)(concat1)
        conv6 = Conv2d_BN(concat1, 512, (3, 3))
        conv6 = Conv2d_BN(conv6, 512, (3, 3))
        conv66 = Conv2dT_BN(conv6, 64, (1, 1), strides=(8, 8))

        convt2 = Conv2dT_BN(conv6, 256, (3, 3))
        concat2 = concatenate([conv3, convt2], axis=3)
        concat2 = Dropout(0.5)(concat2)
        conv7 = Conv2d_BN(concat2, 256, (3, 3))
        conv7 = Conv2d_BN(conv7, 256, (3, 3))
        conv77 = Conv2dT_BN(conv7, 64, (1, 1), strides=(4, 4))

        convt3 = Conv2dT_BN(conv7, 128, (3, 3))
        concat3 = concatenate([conv2, convt3], axis=3)
        concat3 = Dropout(0.5)(concat3)
        conv8 = Conv2d_BN(concat3, 128, (3, 3))
        conv8 = Conv2d_BN(conv8, 128, (3, 3))
        conv88 = Conv2dT_BN(conv8, 64, (1, 1), strides=(2, 2))

        convt4 = Conv2dT_BN(conv8, 64, (3, 3))
        concat4 = concatenate([conv1, convt4], axis=3)
        concat4 = Dropout(0.5)(concat4)
        conv9 = Conv2d_BN(concat4, 64, (3, 3))
        conv99 = Conv2d_BN(conv9, 64, (3, 3))

        out1 = attention_model_B(conv66, 64)
        out2 = attention_model_B(conv77, 64)
        out3 = attention_model_B(conv88, 64)
        out4 = attention_model_B(conv99, 64)

        outZ = concatenate([out1, out2, out3, out4], axis=3)
        outZ = Conv2d_BN(outZ, 64, (1, 1))

        outpt = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(outZ)

        model = Model(inputs=inputs, outputs=outpt)
        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
        return model


    def train(self):
        print("loading data")
        imgs_train, imgs_mask_train, imgs_test = self.load_data()
        print("loading data done")
        model = self.get_unet()
        print("got model")
        model.load_weights('AMCNN.hdf5')
        model_checkpoint = ModelCheckpoint('AMCNN.hdf5', monitor='val_loss',verbose=1, save_best_only=True)
        print('Fitting model...')
        model.fit(imgs_train, imgs_mask_train, batch_size=1, epochs=100, verbose=1, validation_split=0.1, shuffle=True,
                  callbacks=[model_checkpoint])
        print('predict test data')
        imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
        np.save('../results/imgs_mask_test.npy', imgs_mask_test)

    def save_img(self):
        print("array to train")
        imgs = np.load('../results/imgs_mask_test.npy')
        sum = 0.0
        for i in range(imgs.shape[0]):
            img = imgs[i]
            labels = cv2.imread('../Tlabel/%d.png' % (i+980),0)
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