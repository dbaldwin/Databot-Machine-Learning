import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 0 = Left, 1 = Straight, 2 = Right
direction = ['Left', 'Straight', 'Right']

def show_road(img, caption='', subplot=None):
    if subplot==None:
        _,(subplot)=plt.subplots(1,1)
    imgr=img.reshape((10,25))
    print(imgr)
    subplot.axis('off')
    subplot.imshow(imgr, cmap='gray')
    plt.title(caption)


def get_training_data():
    df = pd.read_csv('./driver/training_notebook.csv', header=None)
    road_matrix = df.values

    return road_matrix, df


