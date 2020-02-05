# Pytorch-CartoonGAN
An unofficial Pytorch demo of Paper: CartoonGAN: Generative Adversarial Networks for Photo Cartoonization
(http://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_CartoonGAN_Generative_Adversarial_CVPR_2018_paper.pdf)

## Prerequisites
Code is intended to work with ```Python 3.6.x```, it hasn't been tested with previous versions
For training and testing, the code needs pytorch
For data preparation, the code needs opencv

## Training
### 1. Setup the dataset
First, you will need to download and setup a dataset. 
You can download the dataset from the Baidu net disk: https://pan.baidu.com/s/19FIv4Fjby1knOTXJDWcXBw

Alternatively you can build your own dataset by setting up the following directory structure:

    .
    ├── data                   
    |   ├── train              # Training
    |   |   ├── Cartoon        # Contains Cartoon images, with screenshots of cartoons, you can make by FameCrop.py
    |   |   ├── Cartoon_blur   # Contains Cartoon_blur images, with images in data/train/Cartoon, you can make by edgeDilate.py
    |   |   └── Photo          # Contains Photo images
    |   └── test               # Testing
    |   |   └── B              # Contains test photo images

```
python FameCrop.py
```
This command will randomly crop screenshots(under the *data/train/screenShots* directory) to 256 * 256
```
python edgeDilate.py
```
This command will dilate edges of cartoon images(under the *data/train/Cartoon* directory)
![cartoon image](https://github.com/ty625911724/PyTorch-CartoonGAN-demo/samples/dilate/cartoon.jpg)
![cartoon blur by dilating the edge](https://github.com/ty625911724/PyTorch-CartoonGAN-demo/samples/dilate/cartoon_blur.jpg)

### 2. Train
```
python ./train --dataroot ./data/ --cuda --initialization
```
This command will start a training session of Initialization phase.
After training 10 epochs, it will save model ./output/initial_checkpoint.pth

```
python ./train --dataroot ./data/ --cuda --load_model ./output/initial_checkpoint.pth
```
This command will start a training session of Cartoon GAN.
Both generators and discriminators weights and the will be saved under the output directory, the generated test images will save in the directory output/cartoon_Gen.

If you don't own a GPU remove the --cuda option, although I advise you to get one!

## Testing
```
python ./test --dataroot ./data/ --cuda --load_model ./output/checkpoint100.pth
```
This command will take the images under the *dataroot/test* directory, run them through the generators and save the output under the *output/cartoon_Gen* directories. As with train, some parameters like the weights to load, this can be set by --load_model.

Examples of the generated outputs:

![Real image1](https://github.com/ty625911724/PyTorch-CartoonGAN-demo/samples/cartoon_Gen/1.jpg)
![cartoon1](https://github.com/ty625911724/PyTorch-CartoonGAN-demo/samples/cartoon_Gen/epoch_93_0010.png)
![Real image2](https://github.com/ty625911724/PyTorch-CartoonGAN-demo/samples/cartoon_Gen/3.jpg)
![cartoon2](https://github.com/ty625911724/PyTorch-CartoonGAN-demo/samples/cartoon_Gen/epoch_93_0018.png)
![Real image3](https://github.com/ty625911724/PyTorch-CartoonGAN-demo/samples/cartoon_Gen/6.jpg)
![cartoon3](https://github.com/ty625911724/PyTorch-CartoonGAN-demo/samples/cartoon_Gen/epoch_93_0031.png)

## Acknowledgments
Due to the fact that this is an unofficial implementation, This Code has some problems that have not been solved. The results that have been shown is the best results in epoch 93. The problems are as follows:
1. With the increases of epochs, the results are not always good.
2. The results are not as good as the official predict implementation, especially the extent of simplifying and the lack of clear black edge of generated images.
3. Althought balanced by content loss, it has the problem of model collapse in some epochs.
![example of model Collpase](https://github.com/ty625911724/PyTorch-CartoonGAN-demo/samples/example_model_Collpase.png)
It will be sincerely appreciated if you could help me solve these problems. 

Code is base on https://github.com/aitorzip/PyTorch-CycleGAN