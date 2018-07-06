import os
import pickle
import sys
import codecs
import time
import random
import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import cv2;
import numpy as np
from math import *
import numpy.ma as ma
import random as rand
from typing import List, Set, Dict

from torchloop.util import pipeline_object as po
from torchloop.util import tl_logging
logger = tl_logging.tl_logger()

class picture_pool:
    def __init__(self, bg_dir):
        self._bg_dir = os.path.abspath(bg_dir)
        self._bg_files = os.listdir(bg_dir)

    def sample_bg_file(self, sample_size: int=1) -> List[str]:
        selected_files = rand.sample(self._bg_files, sample_size)
        #### convert into abs path
        selected_files = list(map(lambda x: os.path.join(
            self._bg_dir, x), selected_files))
        return selected_files

class font_pool:
    def __init__(self, font_dir, font_size, file_list=None):
        self._font_dir = os.path.abspath(font_dir)
        if not file_list:
            self._font_files = os.listdir(font_dir)
        else:
            self._font_files = file_list

    def sample_font_file(self, sample_size: int=1) -> List[str]:
        selected_files = rand.sample(self._font_files, sample_size)
        #### convert into abs path
        selected_files = list(map(lambda x: os.path.join(
            self._font_dir, x), selected_files))
        return selected_files

    def font_files(self):
        return self._font_files

def random_scale(x,y):
    ''' 对x随机scale,生成x-y之间的一个数'''
    gray_out = random.randint(x, y)
    return gray_out

def text_gen_gray(bg_gray: int, line):
    gray_flag = np.random.randint(2)
    if bg_gray < line:
        text_gray = random_scale(bg_gray + line, 255)
    elif bg_gray > (255 - line):
        text_gray = random_scale(0, bg_gray - line)
    else:
        text_gray = gray_flag*random_scale(0, bg_gray - line) + (1 - gray_flag)*random_scale(bg_gray+line, 255)
    return text_gray

class font_drawer:
    def __init__(self, font_file: str, bg_file: str, ch_size: int=16, 
            color_margin: int=60):
        self.image_font_ = ImageFont.truetype(font_file, ch_size)
        logger.debug("font file {}".format(font_file))
        self.bg_file = bg_file
        self.color_margin = color_margin
        self.char_size = ch_size

    def draw_sample_gray_bg(self, txt_to_draw: str, 
            if_show: bool=False) -> None:
        '''
            draw chn unicode character on a picture 
            and returns the array representation
        '''
        bg_gray: int = random.randint(0, 255)
        text_gray: int = text_gen_gray(bg_gray, self.color_margin)
        logger.debug("grey bg {} txt grey {}".format(
            bg_gray, text_gray))
        txt_w, txt_h = self.image_font_.getsize(txt_to_draw)
        logger.debug("txt {} size {},{}".format(
            txt_to_draw, txt_w, txt_h))
        '''
            add some padding to img
        '''
        padding_w = random.randint(5, 15)
        padding_h = random.randint(5, 8)
        img_w, img_h = padding_w * 2 + txt_w, padding_h * 2 + txt_h
        pos_w, pos_h = padding_w, padding_h
        '''
            create img with params above 
        '''
        img: Image = Image.new("L", (img_w, img_h), bg_gray)

        draw = ImageDraw.Draw(img)
        draw.text((pos_w, pos_h), txt_to_draw, text_gray, 
                font=self.image_font_)
        if if_show:
            img.show()

    def draw_sample_colored(self, txt_to_draw: str, 
            if_show: bool=False) -> None:
        pass

class runner(po.Irunner):
    def extra_conf(self):
        self.conf = self.args

    def run(self):
        pass
    
