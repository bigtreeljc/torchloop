from easydict import EasyDict as edict
from torchloop.util import config_object as co

# those stuff are torchloop native dataset interface
# it will self def sample schema or use torch native

class dataset_handler(co.configurable):
    def __init__(self, **conf_dic):
        self.conf = edict(conf_dic)

    def handle(self):
        loader_o = self.conf.loader_cls(
                self.conf.dir)
        return loader_o.load()

class Idata_loader(co.configurable):
    # always return Idataset object
    def load():
        raise NotImplementedError

class Idataset(co.configurable):
    def read_data(self):
        raise NotImplementedError

    def batch(self, sample_size):
        raise NotImplementedError

class Isupervised_dataset(Idataset):
    @property
    def training_samples(self):
        raise NotImplementedError

    @property
    def n_labels(self):
        raise NotImplementedError

    @property
    def labels(self):
        raise NotImplementedError

class Iunsupervised_dataset(Idataset):
    pass
