from torchloop.dataset.langs_data import langs_data_dataset_loader
from torchloop.util import config_object as co
from torchloop.util import torchloop_logging
from torchloop.util import device_object as do
import torch.nn as nn
logger = torchloop_logging.getLogger()

#class basic_supervised_trainer(co.configurable):
#    def train_loop(self):
#        self.validate_myself("nn_ops", "dataset", "train_loop")
#        loader_cls = reflection.for_name(self.dataset["loader_cls"])
#        loader_o = loader_cls(self.dataset["dir"])
#        data_set = loader_o.load()
#        loop_worker_cls = reflection.for_name(\
#                self.train_loop["loop_worker_cls"])
#        loop_worker_o = loop_worker_cls(self.train_loop, 
#                dataset_o=data_set)
#        loop_worker_o.loop()

class char_rnn_loop_worker():
    def conf_keys(self):
        return ["nn", "optimizer", "device", "dataset_o"]

    def prepare_parameters(self):
        self.validate_myself(["n_iters", "print_every",
            "matplot", "print_formatter_cls", "plot_every"]) 
        self.device_o = do.torch_device(self.conf.device)
        self.device_o.init_tensor_dtype()
        nn_conf_ = self.nn
        self.m_nn = nn_conf_.nn_cls(nn_conf_.input_dim, 
                nn_conf_.output_dim,
                nn_conf_.hidden_dim)
        self.m_nn = self.device_o.decorate_model(self.m_nn)
        # TODO: load persist nn models
        pass

    def train_step(self, category_tensor, line_tensor):
        self.validate_myself("m_nn")
        criterion = nn.NLLLoss()
        hidden = self.m_nn.initHidden()
        self.m_nn.zero_grad()

        # TODO: make it vectorize
        for i in range(line_tensor.size()[0]):
            output, hidden = self.m_nn(line_tensor[i], hidden)

        loss = criterion(output, category_tensor)
        loss.backward()

        for p in self.m_nn.parameters():
            p.data.add(-learning_rate, p.grad.data)

        return output, loss.item()

    def train_loop(self):
        self.validate_myself(["dataset_o"])
        self.dataset_o
