import torch
import torch.nn as nn
from torch.autograd import Variable
from omegaconf import ListConfig
import warnings
# adapted from: https://github.com/dcodrut/weather2land/blob/main/code/src/models/pt_convlstm/model/conv_lstm/ConvLSTM.py

class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias, use_bn):
        """
        Initialize ConvLSTM cell.
        
        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        use_bn: bool
            Whether or not to use BatchNormalization after each layer.
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]
        self.kernel_size = kernel_size
        self.bias = bias

        conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                         out_channels=4 * self.hidden_dim,
                         kernel_size=self.kernel_size,
                         padding='same',
                         bias=self.bias)

        if not use_bn:
            self.conv = conv
        else:
            self.conv = nn.Sequential(conv, nn.BatchNorm2d(conv.out_channels))

    def forward(self, input_tensor, cur_state, return_gates=False):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        if return_gates:
            return h_next, c_next, torch.stack((i, f, o, g), dim=1)
        return h_next, c_next

    def init_hidden(self, batch_size):
        v1 = Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width))
        v2 = Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width))
        if torch.cuda.is_available():
            v1 = v1.cuda()
            v2 = v2.cuda()
        return v1, v2

class ConvLSTM(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 bias=True, use_bn=False, dropout=0., final_module=None, return_sequence=False):
        super(ConvLSTM, self).__init__()

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * num_layers
        if isinstance(kernel_size, (list,ListConfig)) and len(kernel_size) == 2 and all(isinstance(i, int) for i in kernel_size):
            # is list of 2 integers
            kernel_size = [kernel_size] * num_layers
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim] * num_layers
        elif len(hidden_dim) == 1:
            hidden_dim = hidden_dim * num_layers

        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        if dropout > 0 and num_layers < 2:
            # other posibility for dropout is input and hidden: https://github.com/josephdviviano/lstm-variational-dropout/blob/master/model.py
            warnings.warn('dropout is not applied to the last layer, since it is only one layer, it is not applied.')

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.bias = bias
        self.use_bn = use_bn
        self.dropout = dropout
        self.final_module = final_module
        self.return_sequence = return_sequence

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            ks = self.kernel_size[i]
            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=ks,
                                          bias=self.bias,
                                          use_bn=self.use_bn))

        self.cell_list = nn.ModuleList(cell_list)
        self.dropout_list = nn.ModuleList([nn.Dropout2d(p=self.dropout) for _ in range(self.num_layers - 1)])

    def forward(self, input_tensor, hidden_state=None, return_gates=False):
        """use all input tensors time steps to make a prediction."""
        b, _t, c, h, w = input_tensor.shape
        if hidden_state is None:# init hidden
            hidden_state = self._init_hidden(b)

        preds = [] # loop through time steps
        for t in range(_t):
            inputs = input_tensor[:, t, ...]

            res = self.one_step(inputs, hidden_state, return_gates=return_gates)
            if return_gates:
                h_last, hidden_state, gates = res
                internal_info['states'].append([(x[0].cpu(), x[1].cpu()) for x in hidden_state])
                internal_info['gates'].append(gates)
            else:
                h_last, hidden_state = res
            preds.append(h_last)
        
        if self.final_module is not None: # apply final module e.g. linear layer
            for i in range(len(preds)):
                preds[i] = self.final_module(preds[i])
        # accumulate the predictions per time step
        preds = torch.stack(preds, dim=1)

        if return_gates:
            return h_last, hidden_state, internal_info
        elif self.return_sequence:
            return preds
        else:
            return preds[:,-1,...]

    def one_step(self, input_tensor, hidden_state, return_gates=False):
        """
        make one time step prediction.
        Returns
        -------
        last_state_list, layer_output
        """
        cur_layer_input = input_tensor
        all_gates = []
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            res = self.cell_list[layer_idx](cur_layer_input, cur_state=[h, c], return_gates=return_gates)
            if return_gates:
                h, c, gates = res
                all_gates.append(gates)
            else:
                h, c = res
            hidden_state[layer_idx] = h, c
            cur_layer_input = h

            # apply dropout
            if self.dropout > 0 and layer_idx < self.num_layers - 1:
                cur_layer_input = self.dropout_list[layer_idx](cur_layer_input)

        if return_gates:
            return h, hidden_state, all_gates
        
        return h, hidden_state

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states