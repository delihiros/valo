import cv2
import numpy as np
import argparse
from IPython import display
from matplotlib import pyplot as plt
from ipywidgets import widgets, fixed


def imshow(img, format=".jpg", **kwargs):
    """ndarray 配列をインラインで Notebook 上に表示する。
    """
    img = cv2.imencode(format, img)[1]
    img = display.Image(img, **kwargs)
    display.display(img)

def inRange(img, c1, c2, c3):
    """2値化処理を行い、結果を表示する。
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([c1[0], c2[0], c3[0]])
    upper = np.array([c1[1], c2[1], c3[1]])

    bin_img = cv2.inRange(hsv, lower, upper)
    imshow(bin_img)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required=True, help='path to input')
    args = vars(ap.parse_args())

    names = ["H", "S", "V"]
    parts = {}
    for i, name in enumerate(names, 1):
        slider = widgets.SelectionRangeSlider(
            options=np.arange(256), index=(0, 255), description=name
        )
        slider.layout.width = "400px"
        parts[f"c{i}"] = slider
    
    image = cv2.imread(args['image'])
    widgets.interactive(inRange, **parts, img=fixed(image))
