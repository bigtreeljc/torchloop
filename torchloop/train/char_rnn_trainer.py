from torchloop.dataset.langs_data import langs_data_dataset_loader
from torchloop.util import config_object as co
from torchloop.util import torchloop_logging
from torchloop.util import device_object as do
from torchloop.util import rand_utils, tensor_utils
import torch.nn as nn
logger = torchloop_logging.getLogger()

class char_rnn_loop_worker():
    def conf_keys(self):
        return ["nn", "optimizer", "device", "dataset_o"]

    def prepare_parameters(self):
        self.validate_myself(["n_iters", "print_every",
            "matplot", "print_formatter_cls", "plot_every"]) 
        # prepare nn and device
        self.device_o = do.torch_device(self.conf.device)
        self.device_o.init_tensor_dtype()
        nn_conf_ = self.nn
        self.m_nn = nn_conf_.nn_cls(nn_conf_.input_dim, 
                nn_conf_.output_dim,
                nn_conf_.hidden_dim)
        self.m_nn = self.device_o.decorate_model(self.m_nn)
        # prepare dataset
        # TODO: load persist nn models
        # needs naming stretegy and persister object

    def train_step(self, label_tensor, line_tensor):
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
        self.validate_myself(["dataset_o", "n_iters", "print_every",
            "plot_every", "log_format_cls"])
        for iter_ in range(1, 1 + self.n_iters):
            label, line, label_tensor, line_tensor = \
                    self.dataset_o.batch()
            output, loss = self.train(label_tensor, line_tensor)
            current_loss += loss
            
            if iter_ % print_every == 0:
                guess, guess_i = tensor_utils.label_from_output(
                        output, self.dataset_o.labels)
                correct = '✓' if guess == category else '✗ (%s)' % category
                # TODO: make formatter cls available
                logger.info('%d %d%% (%s) %.4f %s / %s %s' % (
                    iter_, iter_ / n_iters * 100, timeSince(start), 
                    loss, line, guess, correct))

            if iter % plot_every == 0:
                all_losses.append(current_loss / plot_every)
                current_loss = 0
