import unittest
import os
from torchloop.dataset import langs_char_dataset
from torchloop.util import fs_utils
from torchloop.util import tl_logging
logger = tl_logging.tl_logger(tl_logging.DEBUG, True)

# replace with your own dir
langs_dir = os.path.join(fs_utils.src_home(), "data", "langs_data", "names")

class test(unittest.TestCase):
    def test1(self):
        ds_o = langs_char_dataset.langs_char_dataset(dir=langs_dir)
        ds_o.read_data()  

    def test2(self):
        ds_o = langs_char_dataset.langs_char_dataset(
                dir=langs_dir, batch_size=32)
        ds_o.read_data()
        label, line, lt, ct = ds_o.batch()
        logger.info("label {}, lines {}, lt {}, ct {}".format(
            label, line, lt, ct))

    def test3(self):
        ds_o = langs_char_dataset.langs_char_dataset(
                dir=langs_dir, batch_size=32)
        ds_o.read_data()
        n_labels = ds_o.n_labels
        labels = ds_o.labels
        logger.info("n_labels {}, labels {}".format(
            n_labels, labels))

if __name__ == "__main__":
    unittest.main()
