from torchloop.util import tl_logging
logger = tl_logging.tl_logger()
try:
    import cv2
except:
    logger.error("package cv2 not in env")
import matplotlib.pyplot as plt
import torchvision
import numpy as np
from typing import List, Dict, Tuple

def im_show_plt(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def im_toggle_show_plt():
    plt.show()

def make_img_grid(imgs):
    return torchvision.utils.make_grid(imgs)

####
# interfaces for image plotter
####
class I_cv_plotter:
    @staticmethod
    def img_to_show(imgs) -> None:
        raise NotImplementedError        

    @staticmethod
    def toggle_show() -> None:
        raise NotImplementedError

class plt_plotter(I_cv_plotter):
    @staticmethod
    def img_to_show(imgs) -> None:
        ####
        # interactive mode on
        ####
        plt.ion()
        im_show_plt(make_img_grid(imgs))

    @staticmethod
    def toggle_show() -> None:
        plt.ioff()
        plt.show()
