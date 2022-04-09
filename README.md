# AMCNN
Attention-modulated multi-branch convolutional neural networks for neonatal brain tissue segmentation

![5c243482c0108208f94c4f7e5b30bac](https://user-images.githubusercontent.com/16028075/162456676-7136c55d-b80d-4cdf-bb53-61c99a464454.jpg)

- The UNet model framework is widely used in the field of medical image segmentation. Our model is innovated and improved on its basis to increase the accuracy of image segmentation. It has strong robustness and portability.

- Our model is used for neonatal brain tissue segmentation (white matter, gray matter and cerebrospinal fluid). 
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

We have encapsulated each module,

- You need to put the data set in the train and label folder under the deform folder, then call data.Py (data processing file) processes image data into NPY format data (you can rewrite the data processing file according to your needs).
```bash
AMCNN/models/process/data_process.py
```


- You can train the model through the following path.
```bash
AMCNN/models/model_train/models.py
```

- You can test the model through the following path.
```bash
AMCNN/models/model_train/test_models.py
```

## Citation

If you find our code or paper useful, please cite the paper:
```bash
@article{XunliFan2022,
title = {Attention-modulated multi-branch convolutional neural networks for neonatal brain tissue segmentation},
author = {Xunli Fan, Shixi Shan, Xianjun Li, Jinhang Li, Jizong Mi, Jian Yang, Yongqin Zhang},
journal = {Computers in Biology and Medicine, 2022 (Revision)},
volume = {},
pages = {},
year={2022 (Revision)}
}
```
