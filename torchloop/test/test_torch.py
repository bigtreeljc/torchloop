import unittest
import torch.nn as nn
import torch
from captcha.image import ImageCaptcha
import glob
from torchloop.util import tl_logging
import torchvision
logger = tl_logging.tl_logger(tl_logging.DEBUG, True)

def captcha_gen(self):
    font_home = "/home/bigtree/Downloads/open-sans/*ttf"
    image = ImageCaptcha(fonts=glob.glob(font_home))
    
    data = image.generate('AABHEO')
    image.write('AABHEO', 'out.png')

class strLabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=True):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        if isinstance(text, str):
            text = [
                self.dict[char.lower() if self._ignore_case else char]
                for char in text
            ]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts



class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn: int, nHidden: int, nOut: int):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output

class test(unittest.TestCase):
    def test1(self):
        n_input: int = 10
        n_hidden: int = 20
        n_layers: int = 1
        n_batch: int = 16
        width, height = 10, 10
        seq_len = channel = 3              # means seq length
        encoded_length = n_input
        ######
        # input shape [batch_size, seq_len, n_features]
        ######
        bi_lstm1 = nn.LSTM(n_input, n_hidden, n_layers, bidirectional=True)
        mocked_input = torch.randn(n_batch, channel, encoded_length)
        hidden, cell_state = bi_lstm1(mocked_input)
        print("hidden shape {}".format(hidden.size()))
        print("len cell_state {}".format(len(cell_state)))
        print("cell shape {} {}".format(
            cell_state[0].size(), cell_state[1].size()))
        #####
        # n_iter 10
        #####
        n_iter = 10
        input_data = mocked_input
        h_prev, c_prev = torch.randn(2, seq_len, n_hidden), \
                torch.randn(2, seq_len, n_hidden)
        ouput_data, (h_prev, c_prev) = bi_lstm1(
                input_data, (h_prev, c_prev))
    
    def test2(self):
        alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
        converter = utils.strLabelConverter(alphabet)

    def test3(self):
        ####
        # test something like get the last layer of res52
        ####
        resnet = torchvision.models.resnet50(pretrained=True)
        logger.debug("resnet\n {}".format(resnet))
        batch_size: int = 64
        n_channel: int = 3
        n_w, n_h = 224, 224
        batch_ = torch.randn(batch_size, n_channel, n_w, n_h)
        outputs = resnet(batch_)
        logger.debug("outputs {}".format(outputs))

    def test4(self):
        ####
        # get vgg16 conv5_3
        ####
        use_cuda = torch.cuda.is_available()
        logger.debug("cuda avail {}".format(use_cuda))
        vgg16 = torchvision.models.vgg16(pretrained=True)
        # logger.debug("vgg16 keys {}".format(list(vgg16.children())))
        modules = vgg16._modules
        logger.debug("vgg 16\n{}".format(modules.keys()))
        logger.debug("vgg 16\n{}\ntype {}".format(
            modules['features'], type(modules['features'])))
        input_batch = torch.randn(16, 3, 224, 224)
           
        features_getter = modules['features']
        if use_cuda:
            input_batch = input_batch.cuda()
            features_getter = features_getter.cuda()
         
        features_ = features_getter(input_batch)
        logger.debug("features shape {}".format(features_.shape))
        logger.debug("feature getters {}".format(features_getter))
        modulelist = list(vgg16.features.modules())
        input_batch = torch.randn(16, 3, 224, 224)

        if use_cuda:
            input_batch = input_batch.cuda()
            modulelist = features_getter.cuda()
         
        for l in modulelist[:12]:
            input_batch = l(input_batch)
        logger.debug("after conv5 features shape {}".format(input_batch.shape))

    def test4_1(self):
        ####
        # get vgg16 conv5_3
        ####
        use_cuda = torch.cuda.is_available()
        logger.debug("cuda avail {}".format(use_cuda))
        vgg16 = torchvision.models.vgg16(pretrained=True)
        # logger.debug("vgg16 states \n{}".format(vgg16.state_dict()))
        logger.debug("vgg16 keys\n{}".format(vgg16.state_dict().keys()))
        logger.debug("vgg16 modules\n{}".format(list(vgg16.modules())))


    def test5(self):
        #######
        # try out some deconv stuff
        # transform a give batch of 
        # rand 100 dim variable into a img
        #######
        batch_size: int = 64
        n_channel: int = 3
        n_w, n_h = 224, 224
        z_dim = 100
        # batch_ = torch.randn(batch_size, n_channel, n_w, n_h)
        batch_ = torch.randn(batch_size, z_dim)
        
        ######
        # here we go
        ######
        batch_ = batch_.view(batch_size, z_dim, 1, 1)
        logger.debug("batch shape {}".format(batch_.size()))
        deconv1 = nn.ConvTranspose2d(z_dim, 80, 
                kernel_size=3, stride=1, padding=0,
                output_padding=0, bias=False)
        deconved = deconv1(batch_)
        logger.debug("batch shape {}".format(deconved.size()))
        deconv2 = nn.ConvTranspose2d(80, 40, kernel_size=5,
                stride=2, padding=0, bias=False)
        deconved = deconv2(deconved)
        logger.debug("batch shape {}".format(deconved.size()))
        deconv3 = nn.ConvTranspose2d(40, 20, kernel_size=9,
                stride=2, padding=0, bias=False)
        deconved = deconv3(deconved)
        logger.debug("batch shape {}".format(deconved.size()))
        deconv4 = nn.ConvTranspose2d(20, 10, kernel_size=4,
                stride=1, padding=0, bias=False)
        deconved = deconv4(deconved)
        logger.debug("batch shape {}".format(deconved.size()))
        deconv5 = nn.ConvTranspose2d(10, 3, kernel_size=5,
                stride=1, padding=0, bias=False)
        deconved = deconv5(deconved)
        logger.debug("batch shape {}".format(deconved.size()))
        #####
        # success
        #####

    def test6(self):
        ######
        # crf 
        ######
        pass
        
        
if __name__ == "__main__":
    unittest.main()

