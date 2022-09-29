# XRD_denoising
X-ray diffraction denoising using deep convolutional neural networks.

This repository contains Python source code for denoising experimental low-counting statistics X-ray diffraction data. It provides the neural-network definitions as well as a training pipeline. It doesn't contain the code for data pre-processing.

![figure_1_v13](https://user-images.githubusercontent.com/43796543/193001970-d124f3a4-a905-4493-9f18-f1fb9528ddea.png)

Required packages (Python 3):
- numpy
- pandas
- pillow
- scipy
- tensorflow (optional but recommended: tensorflow-gpu)
- nomkl

Original work and data:
- J. Oppliger et al., “Weak-signal extraction enabled by deep-neural-network denoising of diffraction data.” arXiv, Sep. 19, 2022. doi: 10.48550/arXiv.2209.09247.
- J. Oppliger et al., “X-ray diffraction dataset for experimental noise filtering.” Zenodo, Jul. 24, 2022. doi: 10.5281/zenodo.6674077.
