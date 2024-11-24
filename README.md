# Image Alignment
Given a grayscale image of three channels (R, G, B), the goal is to crop this image into 3 channel images and align them to create a color image. 
## Requirements
```python
matplotlib==3.8.4
numpy==2.1.1
Pillow==10.4.0
skimage==0.0
torch==2.4.0
torchvision==0.19.0
```

# 1 Image Alignment by Minimizing Loss (Brute Force)
## How to run

### run all images with each metrics using Green as reference
```python
python main.py -i all
```

### run specific image with specific metric and specific reference color
```python
python main.py -i [image name] -m [metric] -c [color]
ex) python main.py -i 1 -m ncc -c red
```
- image name: 1 to 7
- metric: {mse, ncc, ssim}
- color: {green, red, blue}

### change padding method
- paddings: {circular, zero}

# 2 Image Alignment with Spatial Transformer Network [paper](https://arxiv.org/pdf/1506.02025)
## How to run

### run all images with each metrics
```python
python main_p1.py -i all
```

### run specific image with specific metric
```python
python main_STN.py -i [image name] -m [metric]
ex) python main_STN.py -i 1 -m ncc
```
- image name: 1 to 6
- metric: {mse, ncc}
