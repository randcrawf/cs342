import torch
import torch.nn.functional as F

class ClassificationLoss(torch.nn.Module):
    def forward(self, input, target):
        """
        Your code here

        Compute mean(-log(softmax(input)_label))

        @input:  torch.Tensor((B,C))
        @target: torch.Tensor((B,), dtype=torch.int64)

        @return:  torch.Tensor((,))

        Hint: Don't be too fancy, this is a one-liner
        """
        return F.cross_entropy(input, target)

class CNNClassifier(torch.nn.Module):
    class Block(torch.nn.Module):
        #https://www.programcreek.com/python/example/107671/torch.nn.BatchNorm2d
        def __init__(self, n_input, n_output, stride=1):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(n_input, n_output, kernel_size=3, stride=stride, padding=1),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(),
                torch.nn.Conv2d(n_output, n_output, kernel_size=3, stride=stride, padding=1),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU()
            )

        def forward(self, x):
            return self.net(x)

    def __init__(self):
        """
        Your code here
        """
        super().__init__()

        c = 64
        l = [torch.nn.Conv2d(3, c, 3, padding=1), torch.nn.ReLU()]
        strides = [1,2,1,1,1,2]

        for s in strides:
            l.append(self.Block(c, c, stride=s))

        self.feature_extractor = torch.nn.Sequential(*l)
        self.linear = torch.nn.Linear(c, 10)

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        x[:, :] = x[:, :] * 2 - 1

        x = self.feature_extractor(x).mean(3).mean(2)
        return self.linear(x)


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, CNNClassifier):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'cnn.th'))
    raise ValueError("model type '%s' not supported!"%str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = CNNClassifier()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'cnn.th'), map_location='cpu'))
    return r
