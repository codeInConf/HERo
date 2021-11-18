import torch
import torch.nn as nn

class RST(nn.Module):
    def __init__(self, rnn, fc, relu, RST_depth, cgf, n_layers, rst_bidirect, rst_input_dim):
        super(RST, self).__init__()
        self.rnn = rnn
        self.fc = fc
        self.relu = relu
        self.depth = RST_depth
        self.cgf = cgf
        self.n_layers = n_layers
        self.rst_bidirectional = rst_bidirect
        self.rst_input_dim = rst_input_dim

    # here for handling RST
    def post_order_traversal_RST(self, root, bodytext, layer, device):
        left_embed = None
        right_embed = None
        h_left = None
        h_right = None
        c_left = None
        c_right = None
        if root.label() == 'EDU':
            if int(root[0]) in bodytext:
                out = self.cgf.forward(bodytext[int(root[0])][0], layer, device)
            else:
                out = torch.zeros_like(torch.empty(1, 1, self.rst_input_dim )).to(device)

            if self.rst_bidirectional:
                h = torch.zeros(2*self.n_layers, 1, self.rst_input_dim//2).to(device)
            else:
                h = torch.zeros(self.n_layers, 1, self.rst_input_dim).to(device)
            return out, h
        if root[0] is not None:
            left_embed, h_left = self.post_order_traversal_RST(root[0], bodytext,  layer + 1, device)
        if root[1] is not None:
            right_embed, h_right = self.post_order_traversal_RST(root[1], bodytext,  layer + 1, device)
        if layer <= self.depth:

            h = torch.mean(torch.stack([h_left, h_right],0), dim=0 )

            inpt = torch.cat((left_embed,right_embed),0)

            out, h = self.rnn(inpt, h)
            out = torch.mean(out, 0, True)
        else:
            out = torch.mean(torch.cat((left_embed,right_embed), 0), 0, True)

            if self.rst_bidirectional:
                h = torch.zeros(2*self.n_layers, 1, self.rst_input_dim//2).to(device)
            else:
                h = torch.zeros(self.n_layers, 1, self.rst_input_dim).to(device)

        return out, h

    def forward(self, root, bodytext, layer, device):
        res, _ = self.post_order_traversal_RST(root, bodytext, layer, device)
        return res

class CGF(nn.Module):
    def __init__(self, rnn, fc, relu, embed_dictionary, n_layers, cgf_bidirectional, cgf_input_dim, depth, max_child_number):
        super(CGF, self).__init__()
        self.rnn = rnn
        self.fc = fc
        self.relu = relu
        self.embed_dict = embed_dictionary
        self.n_layers= n_layers
        self.cgf_bidirectional = cgf_bidirectional
        self.cgf_input_dim = cgf_input_dim
        self.depth = depth
        self.max_child_number = max_child_number

    def post_order_traverse1_CGF(self, root, layer,  device):
        if root is not None:
            if isinstance(root, str):
                if self.cgf_bidirectional:
                    h = torch.zeros(2 * self.n_layers, 1, self.cgf_input_dim // 2).to(device)

                else:
                    h = torch.zeros(self.n_layers, 1, self.cgf_input_dim).to(device)

                root = root.lower()
                if root in self.embed_dict:
                    return torch.Tensor([[self.embed_dict[root]]]).to(device), h
                else:
                    return torch.zeros_like(torch.empty(1, 1, self.cgf_input_dim )).to(device), h
        if layer <= self.depth:
            stack_x = None
            stack_h = []

            for i in range(0, min(len(root),self.max_child_number)):
                x, h = self.post_order_traverse1_CGF(root[i], layer+1,  device)
                if i == 0:
                    stack_x = x
                else:
                    stack_x = torch.cat((stack_x, x), 0)
                stack_h .append(h)

            stack_h = torch.mean(torch.stack(stack_h, 0), dim=0)

            out, h = self.rnn(stack_x, stack_h)

            out = torch.mean(out, dim=0, keepdim=True)

            return out, h
        else:
            stack_x = None
            if self.cgf_bidirectional:
                h = torch.zeros(2 * self.n_layers, 1, self.cgf_input_dim // 2).to(device)
            else:
                h = torch.zeros(self.n_layers, 1, self.cgf_input_dim).to(device)

            for i in range(0, min(len(root), self.max_child_number)):
                x, _ = self.post_order_traverse1_CGF(root[i], layer+1,  device)
                if i == 0:
                    stack_x = x
                else:
                    stack_x = torch.cat((stack_x, x), 0)

            stack_x = torch.mean(stack_x, dim=0, keepdim=True)

            return stack_x, h

    def forward(self, root, layer, device):
        res, _ = self.post_order_traverse1_CGF(root, layer, device)

        return res




