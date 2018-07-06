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
from typing import List, Set, Dict, Tuple

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
    def __init__(self, font_dir, file_list=None):
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

def text_gen_colored(bg_color: Tuple, line) -> Tuple:
    txt_color = [0, 0, 0]
    # 3 channels
    assert len(bg_color) == 3

    for c_ind, bg_gray in enumerate(bg_color):
        txt_gray = text_gen_gray(bg_gray, line)
        txt_color[c_ind] = txt_gray

    return tuple(txt_color)
        
class font_drawer:
    def __init__(self, font_file: str, bg_file: str, ch_size: int=16, 
            color_margin: int=60):
        self.image_font_ = ImageFont.truetype(font_file, ch_size)
        logger.debug("font file {}".format(font_file))
        self.bg_file = bg_file
        self.bg = Image.open(bg_file)
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
        bg_color =\
               random.randint(0, 255), random.randint(0, 255), \
               random.randint(0, 255)
        text_color: int = text_gen_colored(bg_color, self.color_margin)
        logger.debug("bg color {} txt color {}".format(
            bg_color, text_color))
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
        img: Image = Image.new("RGB", (img_w, img_h), bg_color)

        draw = ImageDraw.Draw(img)
        draw.text((pos_w, pos_h), txt_to_draw, text_color, 
                font=self.image_font_)
        if if_show:
            img.show()

    def _mk_colors(self, txt_to_draw: str) -> Tuple:
        bg_color =\
               random.randint(0, 255), random.randint(0, 255), \
               random.randint(0, 255)
        text_color: int = text_gen_colored(bg_color, self.color_margin)
        logger.debug("bg color {} txt color {}".format(
            bg_color, text_color))
        return [bg_color, text_color]

    def _mk_size_pos(self, txt_to_draw: str) -> Tuple:
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
        return img_w, img_h, pos_w, pos_h

    def _draw_text_on_img(self, txt_to_draw: str, img: Image,
            pos_w: int, pos_h: int, text_color: Tuple) -> None:
        draw = ImageDraw.Draw(img)
        draw.text((pos_w, pos_h), txt_to_draw, text_color, 
                font=self.image_font_)

    def draw_sample_colored_bg(self, txt_to_draw: str, 
            if_show: bool=False, show_bg: bool=False,
            show_cropped=False) -> None:
        if show_bg:
            logger.debug("showing background picture {}".format(
                self.bg_file))
            self.bg.show()

        colors = self._mk_colors(txt_to_draw)
        bg_color, txt_color = colors[0], colors[1]
        txt_w, txt_h, txt_pos_w, txt_pos_h = \
                self._mk_size_pos(txt_to_draw)

        '''
            select a bbox in the bg picture
        '''
        bg_w, bg_h = self.bg.size
        assert txt_w < bg_w and txt_h < bg_h, "font too large for bg"
        available_area = bg_w - txt_w, bg_h - txt_h
        selected_w, selected_h = random.randint(0, available_area[0]), \
                    random.randint(0, available_area[0])
        area = (selected_w, selected_h, 
                selected_w + txt_w, selected_h + txt_h)
        cropped_img = self.bg.crop(area)
        if show_cropped:
            logger.debug("showing cropped img")
            cropped_img.show()
        self._draw_text_on_img(txt_to_draw, cropped_img, 
                txt_pos_w, txt_pos_h, txt_color)

        if if_show:
            cropped_img.show()



    def obfascate_sample(self):
        pass

class runner(po.Irunner):
    def extra_conf(self):
        self.conf = self.args

    def run(self):
        pass
    
