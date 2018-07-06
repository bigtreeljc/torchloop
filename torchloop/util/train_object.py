from torchloop.dataset.langs_char_dataset import langs_char_dataset
from torchloop.util import config_object as co
from torchloop.util import tl_logging, dataset_object
logger = tl_logging.tl_logger()

class basic_supervised_trainer(co.configurable):
    def train(self):
        self.validate_myself("nn_ops", "dataset", "train_loop")
        loader_cls = reflection.for_name(self.dataset.loader_cls)
        loader_o = loader_cls(self.dataset)
        data_set = loader_o.read_data()
        loop_worker_cls = reflection.for_name(\
                self.train_loop.loop_worker_cls)
        loop_worker_o = loop_worker_cls(self.train_loop, 
                dataset_o=data_set)
        loop_worker_o.loop()

class Isupervised_train_loop_worker(co.configurable):
    def setup_training(self):
        self.validate_myself()
        self.prepare_parameters()

    def prepare_parameters(self):
        raise NotImplementedError

    def conf_keys(self):
        raise NotImplementedError

    def train_step(self):
        raise NotImplementedError

    def train_loop(self):
        raise NotImplementedError
