import unittest
from torchloop.util import fs_utils
from torchloop.util import torchloop_logging
logger = torchloop_logging.tl_logger(torchloop_logging.DEBUG, True)

class test(unittest.TestCase):
    def test1(self):
        example_file = "/aaa/bbb/ccc/ddd.ext" 
        fname_ = fs_utils.file_name(example_file)
        logger.debug("fname is {}".format(fname_))
        self.assertTrue(fname_ == "ddd")

    def test2(self):
        src_home_ = fs_utils.src_home()
        tl_home_ = fs_utils.torchloop_home()
        logger.debug("src {} tl {}".format(src_home_, tl_home_))

if __name__ == "__main__":
    unittest.main()
