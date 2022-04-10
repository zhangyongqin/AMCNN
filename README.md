# AMCNN
Xunli Fan, Shixi Shan, Xianjun Li, Jinhang Li, Jizong Mi, Jian Yang, Yongqin Zhang. Attention-modulated multi-branch convolutional neural networks for neonatal brain tissue segmentation, Computers in Biology and Medicine, 2022 (Revision)

![5c243482c0108208f94c4f7e5b30bac](https://user-images.githubusercontent.com/16028075/162456676-7136c55d-b80d-4cdf-bb53-61c99a464454.jpg)

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
AMCNN/models/process/data_process.py
```


- You can train the model in the following path.
```bash
AMCNN/models/model_train/models.py
```

- You can test the model in the following path.
```bash
AMCNN/models/model_train/test_models.py
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
