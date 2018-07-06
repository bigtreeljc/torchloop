import unittest
import torchloop.dataset.ocr.generate_ocr as go
from torchloop.util import fs_utils, tl_logging
logger = tl_logging.tl_logger(tl_logging.DEBUG, True)

class test(unittest.TestCase):
    def test1(self):
        bg_dir = fs_utils.default_bg_dir()
        pic_pool = go.picture_pool(bg_dir)
        sampled_bg = pic_pool.sample_bg_file(2)
        print("sampled bgs {}".format(sampled_bg))

    def test2(self):
        font_dir = fs_utils.default_font_dir()
        font_pool = go.font_pool(font_dir)
        sampled_font = font_pool.sample_font_file(2)
        print("sampled fonts {}".format(sampled_font))

    def test3(self):
        '''
            some constants
        '''
        n_sampled: int = 1
        txt_to_draw_: str = "你好，李攀"
        ch_h = 64
        '''
            sampling bg
        '''
        bg_dir = fs_utils.default_bg_dir()
        pic_pool = go.picture_pool(bg_dir)
        sampled_bg = pic_pool.sample_bg_file(n_sampled)
        '''
            sampling font
        '''
        font_dir = fs_utils.default_font_dir()
        font_pool = go.font_pool(font_dir)
        sampled_font = font_pool.sample_font_file(n_sampled)
        '''
            testing text drawer
        '''
        bg_file: str = sampled_bg[0]
        font_file: str = sampled_font[0]
        text_drawer_o = go.font_drawer(font_file, bg_file, ch_size=ch_h)
        text_drawer_o.draw_sample_gray_bg(txt_to_draw_, if_show=True)

if __name__ == "__main__":
    unittest.main()
