# AMCNN
Xunli Fan, Shixi Shan, Xianjun Li, Jinhang Li, Jizong Mi, Jian Yang, Yongqin Zhang. Attention-modulated multi-branch convolutional neural networks for neonatal brain tissue segmentation, Computers in Biology and Medicine, 2022 (Revision)

![J)7T@XETYB`P E2V@BE P V](https://user-images.githubusercontent.com/16028075/162601097-7a0661bb-166a-49fc-a9b5-388bfebb9dd8.png)


The main contributions of this paper are summarized as follows:
- A novel lightweight attention-modulated multi-branch encoder-decoder framework is proposed for accurate neonatal brain tissue segmentation;
- The multi-scale dilated convolution modules are introduced in the encoding path for complete feature extraction;
- The multi-branch attention modules are incorporated in the decoding path for high-fidelity feature reconstruction;
- Spatial attention modules are designed to connect the encoding and decoding paths for ensuring cross-scale spatial information propagation.

## Prerequisites 

- NVIDIA GPU + CUDA CuDNN (CPU untested, feedback appreciated) 
- Keras 2.1.5;
  tensorflow-gpu 2.4.1;
  h5py 2.10.0;
  imageio 2.13.1;
  matplotlib 3.3.4;
  numpy 1.19.5; 
  scikit-image 0.17.2
 
## How to run

- You first load data and labels in the separate paths: .../deform/train and .../deform/label, then call data.py for image preprocessing.

```bash
models/process/data_process.py

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
```


- You can train the model in the following path.
```bash
AMCNN/models/model_train/models.py
```

- You can test the model in the following path.
```bash
models/model_train/test_models.py

imgs_train, imgs_mask_train, imgs_test = self.load_data()
print("test_model")
model = self.get_unet()
print("got model")
model.load_weights('AMCNN.hdf5')
model_checkpoint = ModelCheckpoint('AMCNN.hdf5', monitor='val_loss',verbose=1, save_best_only=True)
print('model predict...')
imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
np.save('../results/imgs_mask_test.npy', imgs_mask_test)
```

## Citation

If you find our code or paper useful, please cite the paper:
```bash
@article{XunliFan2022,
title = {Attention-modulated multi-branch convolutional neural networks for neonatal brain tissue segmentation},
author = {Xunli Fan, Shixi Shan, Xianjun Li, Jinhang Li, Jizong Mi, Jian Yang, Yongqin Zhang},
journal = {Computers in Biology and Medicine},
volume = {},
pages = {},
year={2022 (Revision)}
}
```
