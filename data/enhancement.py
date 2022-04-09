

from keras.preprocessing.image import ImageDataGenerator ,array_to_img , img_to_array ,load_img
import  numpy as np
import os
import  glob
import cv2

class myAugmentation(object):
    '''
    图像增强
    '''
    def __init__(self,train_path="train",label_path="label",merge_path="merge",aug_merge_path="aug_merge", aug_train_path="aug_train", aug_label_path="aug_label", img_type="tif"):
        '''
        使用glob从路径中得到所有的.img_type文件，初始化类：__init()

        '''
        self.train_imgs = glob.glob(train_path+"/*."+img_type)
        self.label_imgs = glob.glob(label_path+"/*."+img_type)
        self.train_path = train_path
        self.label_path = label_path
        self.merge_path = merge_path
        self.img_type = img_type
        self.aug_merge_path = aug_merge_path
        self.aug_label_path = aug_label_path
        self.aug_train_path = aug_train_path
        self.slices = len(self.train_imgs)
        self.datagen = ImageDataGenerator(
            rotation_range=0.2,
            width_shift_range=0.05,
            height_shift_range=0.05,
            shear_range=0.05,
            zoom_range=0.05,
            horizontal_flip=True,
            fill_mode='nearest'
        )

    def Augmentation(self):
        trains = self.train_imgs
        labels = self.label_imgs
        path_train = self.train_path
        path_label = self.label_path
        path_merge = self.merge_path
        imgtype = self.img_type
        path_aug_merge = self.aug_merge_path

        if (len(trains) != len(labels) or len(trains) == 0 or len(labels) == 0):
            print("trains can't match labels")
            return 0
        for i in range(len(trains)):
            img_t = load_img(path_train + "/" + str(i) + "." + imgtype)
            img_l = load_img(path_label + "/" + str(i) + "." + imgtype)
            x_t = img_to_array(img_t)
            x_l = img_to_array(img_l)
            x_t[:, :, 2] = x_l[:, :, 0]
            img_tmp = array_to_img(x_t)
            img_tmp.save(path_merge + "/" + str(i) + "." + imgtype)
            img = x_t
            img = img.reshape((1,) + img.shape)
            savedir = path_aug_merge + "/" + str(i)
            if (not os.path.lexists(savedir)):
                os.mkdir(savedir)
            self.doAugmentate(img,savedir,str(i))



    def doAugmentate(self, img, save_to_dir, save_prefix, batch_size=1, save_format='tif', imgunm=30,):
        datagen = self.datagen
        i = 0
        for batch in datagen.flow(img,
                                  batch_size=batch_size,
                                  save_to_dir=save_to_dir,
                                  save_prefix=save_prefix,
                                  save_format=save_format):
            i += 1
            if (i > imgunm):
                break


    def splitMerge(self):
        path_merge = self.aug_merge_path
        path_train = self.aug_train_path
        path_label = self.aug_label_path

        for i in range(self.slices):
            path = path_merge + "/" + str(i)
            train_imgs = glob.glob(path + "/*." + self.img_type)
            savedir = path_train + "/" + str(i)
            if (not os.path.lexists(savedir)):
                os.mkdir(savedir)
            for imgname in train_imgs:
                midname = cv2.imread(imgname)
                img_train = img[:, :, 2]
                img_label = img[:, :, 0]
                cv2.imwrite(path_train + "/" + str(i) + "/" + midname + "_train" + "." + self.img_type, img_train)
                cv2.imwrite(path_label + "/" + str(i) + "/" + midname + "_train" + "." + self.img_type, img_label)

    def splitTransform(self):
        # 拆分透视变换后的图像
        path_merge = "deform/deform_norm2"
        path_train = "deform/train/"
        path_label = "deform/label/"

        train_imgs = glob.glob(path_merge + "/*." + self.img_type)
        for imgname in train_imgs:
            midname = imgname[imgname.rindex("/") + 1:imgname.rindex("." + self.img_type)]
            img = cv2.imread(imgname)
            img_train = img[:, :, 2]  # cv2 read train rgb->bgr
            img_label = img[:, :, 0]
            cv2.imwrite(path_train + midname + "." + self.img_type, img_train)
            cv2.imwrite(path_label + midname + "." + self.img_type, img_label)


class dataProcess(object):
    def __init__(self,out_rows,out_cols, data_path = "../deform/train", label_path = "../deform/label", test_path = "../train", npy_path = "../npydata", img_type = "tif"):
        self.out_rows = out_rows
        self.out_cols = out_cols
        self.data_path = data_path
        self.label_path = label_path
        self.img_type = img_type
        self.test_path = test_path
        self.npy_path = npy_path

    def creat_train_data(self):
        i = 0
        print('-'*30)
        print('Creating training images...')
        print('-'*30)
        imgs = glob.glob(self.data_path+"/*."+self.img_type)
        print(len(imgs))

        imgdatas = np.ndarray((len(imgs), self.out_rows, self.out_cols, 1), dtype=np.uint8)
        imglabels = np.ndarray((len(imgs), self.out_rows, self.out_cols, 1), dtype=np.uint8)
        for imgname in imgs:
            midname = imgname[imgname.rindex("n") + 2:]
            img = load_img(self.data_path + "/" + midname, grayscale=True)
            label = load_img(self.label_path + "/" + midname, grayscale=True)
            img = img_to_array(img)
            label = img_to_array(label)
            # img = cv2.imread(self.data_path + "/" + midname,cv2.IMREAD_GRAYSCALE)
            # label = cv2.imread(self.label_path + "/" + midname,cv2.IMREAD_GRAYSCALE)
            # img = np.array([img])
            # label = np.array([label])
            imgdatas[i] = img
            imglabels[i] = label
            if i % 100 == 0:
                print('Done: {0}/{1} images'.format(i, len(imgs)))
            i += 1
        print('loading done')
        np.save(self.npy_path + '/train.npy', imgdatas)
        np.save(self.npy_path + '/tlabel.npy', imglabels)
        print('Saving to .npy files done.')






































































