# CS59300CVD Assignment 1

## Requirements
```python
matplotlib==3.8.4
numpy==2.1.1
Pillow==10.4.0
skimage==0.0
torch==2.4.0
torchvision==0.19.0
```

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
- paddings: {"circular", "zero"}

