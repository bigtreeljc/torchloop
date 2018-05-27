from torchloop.dataset.langs_data import langs_data_dataset_loader
from torchloop.util import config_object as co
from torchloop.util import torchloop_logging, reflection
import torch.nn as nn
logger = torchloop_logging.getLogger()

class basic_supervised_trainer(co.configurable):
    def train_loop(self):
        self.validate_myself("nn_ops", "dataset", "train_loop")
        loader_cls = reflection.for_name(self.dataset["loader_cls"])
        loader_o = loader_cls(self.dataset["dir"])
        data_set = loader_o.load()
        loop_worker_cls = reflection.for_name(\
                self.train_loop["loop_worker_cls"])
        loop_worker_o = loop_worker_cls(self.train_loop, 
                dataset_o=data_set)
        loop_worker_o.loop()

class char_rnn_loop_worker(co.configurable):
    def train(self, category_tensor, line_tensor):
        self.validate_myself("rnn")
        criterion = nn.NLLLoss()
        hidden = self.rnn.initHidden()
        rnn.zero_grad()

        for i in range(line_tensor.size()[0]):
            output, hidden = self.rnn(line_tensor[i], hidden)

        loss = criterion(output, category_tensor)
        loss.backward()

        for p in self.rnn.parameters():
            p.data.add(-learning_rate, p.grad.data)

        return output, loss.item()

    def loop(self):
        self.validate_myself(["dataset_o"])
