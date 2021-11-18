from module import *

class Net(nn.Module):
    def __init__(self, max_depth, max_number_child, device, embed_dict, cgf_input_dim, cgf_output_dim,
                 cgf_bias, hidden_dim, rst_input_dim, rst_output_dim, n_layers, rst_bidirect=True, cgf_drop_prob=0.2, cgf_bidirect=True):
        super(Net, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.cgf_input_dim = cgf_input_dim
        self.cgf_output_dim = cgf_output_dim
        self.cgf_fc = nn.Linear(self.cgf_input_dim,self.cgf_output_dim)
        self.cgf_relu = nn.LeakyReLU()
        self.cgf_bidirectional = cgf_bidirect
        self.rnn = None

        if self.cgf_bidirectional:
            self.rnn = nn.GRU(self.cgf_input_dim,  self.cgf_input_dim//2, self.n_layers, cgf_bias, dropout=cgf_drop_prob,
                                   bidirectional=cgf_bidirect)
        else:
            self.rnn = nn.GRU(self.cgf_input_dim,  self.cgf_input_dim, self.n_layers, cgf_bias, dropout=cgf_drop_prob,
                                   bidirectional=cgf_bidirect)

        self.cgf = CGF(self.rnn, self.cgf_fc, self.cgf_relu, embed_dict, self.n_layers, self.cgf_bidirectional, self.cgf_input_dim, max_depth, max_number_child)

        self.RST_depth = max_depth
        self.rst_input_dim = rst_input_dim
        self.rst_output_dim = rst_output_dim
        self.rst_fc = nn.Linear(self.rst_input_dim,  self.rst_output_dim)
        self.rst_relu = nn.LeakyReLU()
        self.rst_bidirectional = rst_bidirect


        self.RST = RST(self.rnn, self.rst_fc, self.rst_relu, self.RST_depth, self.cgf, self.n_layers, self.rst_bidirectional,  self.rst_input_dim)
        self.final_fc = nn.Linear(100, 2)
        self.m = nn.Softmax(dim=2)

    def forward(self, root, bodytext):
        out = self.RST.forward(root, bodytext, 0, self.device)
        out = self.m(self.final_fc(out))

        return out
