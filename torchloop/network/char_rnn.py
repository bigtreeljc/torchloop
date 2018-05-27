import torch.nn as nn

class char_rnn_vanilla_single_layer(nn.Module):
    def __init__(self, input_size_, hidden_size_, output_size_):
        super(char_rnn, self).__init__()
        self.hidden_size = hidden_size_

        self.i2h = nn.Linear(input_size_ + hidden_size_, hidden_size_)
        self.i2o = nn.Linear(input_size_ + hidden_size_, output_size_)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_, hidden_):
        combined = torch.cat((input_, hidden_), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
