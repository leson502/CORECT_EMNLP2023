import torch
import torch.nn as nn

import torch.nn.functional as F

import corect

log = corect.utils.get_logger()


class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_size, tag_size, args):
        super(Classifier, self).__init__()
        self.args = args
        if args.use_highway:
            self.highway = Highway(size=input_dim, num_layers=1, f=F.relu)
            print("*******Using  Highway*******")
        else:
            self.highway = nn.Identity()
        self.lin1 = nn.Linear(input_dim, hidden_size)
        self.drop = nn.Dropout(args.drop_rate)
        self.lin2 = nn.Linear(hidden_size, tag_size)
        self.lin_7 = nn.Linear(hidden_size, 7)
        self.linear = nn.Linear(input_dim, tag_size)
        if args.class_weight:
            if args.dataset == "iemocap":
                self.loss_weights = torch.tensor(
                    [
                        1 / 0.086747,
                        1 / 0.144406,
                        1 / 0.227883,
                        1 / 0.160585,
                        1 / 0.127711,
                        1 / 0.252668,
                    ]
                ).to(args.device)
            elif args.dataset == "iemocap_4":
                self.loss_weights = torch.tensor(
                    [
                        1 / 0.1426370239929562,
                        1 / 0.2386088487783403,
                        1 / 0.37596302003081666,
                        1 / 0.24279110719788685,
                    ]
                ).to(args.device)
        
            self.nll_loss = nn.NLLLoss(self.loss_weights)
            print("*******weighted loss*******")
        else:
            self.nll_loss = nn.NLLLoss()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="sum")

    def get_prob(self, h, text_len_tensor):
        if self.args.use_highway:
            h = self.highway(h)
        hidden = self.drop(F.relu(self.lin1(h)))
        if self.args.emotion == "7class":
            scores = self.lin_7(hidden)
        else:
            scores = self.lin2(hidden)
        log_prob = F.log_softmax(scores, dim=1)
        return log_prob

    def forward(self, h, text_len_tensor):
        log_prob = self.get_prob(h, text_len_tensor)
        y_hat = torch.argmax(log_prob, dim=-1)
        return y_hat

    def get_loss(self, h, label_tensor, text_len_tensor):
        log_prob = self.get_prob(h, text_len_tensor)
        loss = self.nll_loss(log_prob, label_tensor)
        return loss


class Highway(nn.Module):
    def __init__(self, size, num_layers, f):

        super(Highway, self).__init__()

        self.num_layers = num_layers

        self.nonlinear = nn.ModuleList(
            [nn.Linear(size, size) for _ in range(num_layers)]
        )

        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

        self.f = f

    def forward(self, x):
        """
            :param x: tensor with shape of [batch_size, size]
            :return: tensor with shape of [batch_size, size]
            applies σ(x) ⨀ (f(G(x))) + (1 - σ(x)) ⨀ (Q(x)) transformation | G and Q is affine transformation,
            f is non-linear transformation, σ(x) is affine transformation with sigmoid non-linearition
            and ⨀ is element-wise multiplication
            """

        for layer in range(self.num_layers):
            gate = torch.sigmoid(self.gate[layer](x))

            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)

            x = gate * nonlinear + (1 - gate) * linear

        return x
