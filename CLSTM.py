import torch
import torch.nn as nn
from torch.autograd import Variable

# Batch x NumChannels x Height x Width
# UNET --> BatchSize x 1 (3?) x 240 x 240
# BDCLSTM --> BatchSize x 64 x 240 x240

''' Class CLSTMCell.
    This represents a single node in a CLSTM series.
    It produces just one time (spatial) step output.
'''


class CLSTMCell(nn.Module):

    # Constructor
    def __init__(self, input_channels, hidden_channels,
                 kernel_size, bias=True):
        super(CLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.bias = bias
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(self.input_channels + self.hidden_channels,
                              self.num_features * self.hidden_channels,
                              self.kernel_size,
                              1,
                              self.padding)

    # Forward propogation formulation
    def forward(self, x, h, c):
        # print('x: ', x.type)
        # print('h: ', h.type)
        combined = torch.cat((x, h), dim=1)
        A = self.conv(combined)

        # NOTE: A? = xz * Wx? + hz-1 * Wh? + b? where * is convolution
        (Ai, Af, Ao, Ag) = torch.split(A,
                                       A.size()[1] // self.num_features,
                                       dim=1)

        i = torch.sigmoid(Ai)     # input gate
        f = torch.sigmoid(Af)     # forget gate
        o = torch.sigmoid(Ao)     # output gate
        g = torch.tanh(Ag)

        c = c * f + i * g           # cell activation state
        h = o * torch.tanh(c)     # cell hidden state

        return h, c

    @staticmethod
    def init_hidden(batch_size, hidden_c, shape):
        try:
            return(Variable(torch.zeros(batch_size,
                                    hidden_c,
                                    shape[0],
                                    shape[1])).cuda(),
               Variable(torch.zeros(batch_size,
                                    hidden_c,
                                    shape[0],
                                    shape[1])).cuda())
        except:
            return(Variable(torch.zeros(batch_size,
                                    hidden_c,
                                    shape[0],
                                    shape[1])),
                    Variable(torch.zeros(batch_size,
                                    hidden_c,
                                    shape[0],
                                    shape[1])))


''' Class CLSTM.
    This represents a series of CLSTM nodes (one direction)
'''


class CLSTM(nn.Module):
    # Constructor
    def __init__(self, input_channels=64, hidden_channels=[64],
                 kernel_size=5, bias=True):
        super(CLSTM, self).__init__()

        # store stuff
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)

        self.bias = bias
        self.all_layers = []

        # create a node for each layer in the CLSTM
        for layer in range(self.num_layers):
            name = 'cell{}'.format(layer)
            cell = CLSTMCell(self.input_channels[layer],
                             self.hidden_channels[layer],
                             self.kernel_size,
                             self.bias)
            setattr(self, name, cell)
            self.all_layers.append(cell)

    # Forward propogation
    # x --> BatchSize x NumSteps x NumChannels x Height x Width
    #       BatchSize x 2 x 64 x 240 x 240
    def forward(self, x):
        bsize, steps, _, height, width = x.size()
        internal_state = []
        outputs = []
        for step in range(steps):
            input = torch.squeeze(x[:, step, :, :, :], dim=1)
            for layer in range(self.num_layers):
                # populate hidden states for all layers
                if step == 0:
                    (h, c) = CLSTMCell.init_hidden(bsize,
                                                   self.hidden_channels[layer],
                                                   (height, width))
                    internal_state.append((h, c))

                # do forward
                name = 'cell{}'.format(layer)
                (h, c) = internal_state[layer]

                input, c = getattr(self, name)(
                    input, h, c)  # forward propogation call
                internal_state[layer] = (input, c)

            outputs.append(input)

        #for i in range(len(outputs)):
        #    print(outputs[i].size())
        return outputs


class BDCLSTM(nn.Module):
    # Constructor
    def __init__(self, input_channels=64, hidden_channels=[64],
                 kernel_size=5, bias=True, num_classes=2):

        super(BDCLSTM, self).__init__()
        self.forward_net = CLSTM(
            input_channels, hidden_channels, kernel_size, bias)
        self.reverse_net = CLSTM(
            input_channels, hidden_channels, kernel_size, bias)
        self.conv = nn.Conv2d(
            2 * hidden_channels[-1], num_classes, kernel_size=1)
        self.soft = nn.Softmax2d()

    # Forward propogation
    # x --> BatchSize x NumChannels x Height x Width
    #       BatchSize x 64 x 240 x 240
    def forward(self, x1, x2, x3):
        x1 = torch.unsqueeze(x1, dim=1)
        x2 = torch.unsqueeze(x2, dim=1)
        x3 = torch.unsqueeze(x3, dim=1)

        xforward = torch.cat((x1, x2), dim=1)
        xreverse = torch.cat((x3, x2), dim=1)

        yforward = self.forward_net(xforward)
        yreverse = self.reverse_net(xreverse)

        # assumes y is BatchSize x NumClasses x 240 x 240
        # print(yforward[-1].type)
        ycat = torch.cat((yforward[-1], yreverse[-1]), dim=1)
        # print(ycat.size())
        y = self.conv(ycat)
        # print(y.type)
        y = self.soft(y)
        # print(y.type)
        return y
