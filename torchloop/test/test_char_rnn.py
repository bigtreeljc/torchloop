import unittest
from torchloop.network import char_rnn 
from torchloop.dataset import langs_char_dataset
from torchloop.util import nlp_utils, fs_utils
import os
import torch
from torchloop.util import tl_logging
logger = tl_logging.tl_logger(tl_logging.DEBUG, True)

class test(unittest.TestCase):
    def test1(self):
        logger.info("only test the vannila rnn implementation")
        all_letters = nlp_utils.all_letters
        langs_dir = os.path.join(fs_utils.src_home(), "data", "langs_data", "names")
        ds_o = langs_char_dataset.langs_char_dataset(dir=langs_dir)
        ds_o.read_data()  
        input_size = len(all_letters)
        hidden_size = n_hidden = 128
        output_size = ds_o.n_labels
        test_str = "Albert"
        input_ = nlp_utils.lineToTensor(test_str)
        rnn = char_rnn.char_rnn_vanilla_single_layer(
                input_size, hidden_size, output_size)
        hidden = torch.zeros(1, n_hidden)
        output, next_hodden = rnn(input_[0], hidden)
        logger.debug("output {} next states {}".format(
            output, hidden))
        logger.debug("output shape {} next hidden shape {}".format(
            output.size(), hidden.size()))

if __name__ == "__main__":
    unittest.main()
