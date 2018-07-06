import unittest
from torchloop.pipeline import char_rnn_pipeline 
from torchloop.util import fs_utils
import os

class test(unittest.TestCase):
    def test1(self):
        conf_file = os.path.join(
                fs_utils.default_conf_dir(), 
                "char_rnn_default_conf.yml") 
        pipeline_o = char_rnn_pipeline.char_rnn_pipeline(conf_file)
        pipeline_o.run()

if __name__ == "__main__":
    unittest.main()
