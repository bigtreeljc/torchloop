import os
from torchloop.util import dataset_object
from torchloop.util import nlp_utils
from torchloop.util import fs_utils
from torchloop.util import rand_utils
from torchloop.util import tl_logging
logger = tl_logging.tl_logger()

class langs_char_dataset(dataset_object.Isupervised_dataset):
    def read_data(self):
        self.validate_myself(["dir"])
        logger.info("reading dir {}, class {}".format(
            self.dir, __class__))
        self.labels_ = []
        self.label_lines = {}
        file_glob = os.path.join(self.dir, "*.txt")

        for fname in fs_utils.find_files(file_glob):
            label = fs_utils.file_name(fname)
            self.labels_.append(label)
            lines = fs_utils.read_lines(fname)
            self.label_lines[label] = lines

        self.n_labels_ = len(self.labels_)
        logger.debug("n labels is {}".format(
            self.n_labels))
        logger.debug("first 2 of the chinese is {}".format(
            self.label_lines["Chinese"][:2]))
        logger.info("done reading labels {}".format(
            self.labels))
        
    # return the one line tensor and 
    # one label one-hot tensor
    # TODO: make it a real batch, thus we can achieve better speed 
    # with our resources
    def batch(self): 
        self.validate_myself("batch_size") 
        return rand_utils.randomTrainingExample(
                self.labels, self.label_lines)

    @property
    def training_samples(self):
        return self.label_lines

    @property
    def n_labels(self):
        return self.n_labels_

    @property
    def labels(self):
        return self.labels_
