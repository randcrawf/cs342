import torch
import torch.nn.functional as F

#Sources:
#http://www.philkr.net/cs342/lectures/making_it_work/18.html (ResNets)
#https://github.com/philkr/cat_vs_dog

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
        def __init__(self, n_input, n_output, stride=1):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(n_input, n_output, kernel_size=3, stride=stride, padding=1, bias=False),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(),
                torch.nn.Conv2d(n_output, n_output, kernel_size=3, padding=1, bias=False),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU()
            )
            
        def forward(self, x):
            return self.net(x)

    def __init__(self):
        super().__init__()
        """
        Your code here
        Hint: Base this on yours or HW2 master solution if you'd like.
        Hint: Overall model can be similar to HW2, but you likely need some architecture changes (e.g. ResNets)
        """
        c = 32
        l = [torch.nn.Conv2d(3, c, kernel_size=7, padding=3, stride=2, bias=False), torch.nn.BatchNorm2d(c), torch.nn.ReLU(), torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        layers = [32, 64, 128]
        for layer in layers:
            l.append(self.Block(c, layer, stride=2))
            c = layer

        self.feature_extractor = torch.nn.Sequential(*l)
        self.classifier = torch.nn.Linear(c, 10)

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        """
        x = self.feature_extractor(x).mean(3).mean(2)
        return self.classifier(x)


class FCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """
        Your code here.
        Hint: The FCN can be a bit smaller the the CNNClassifier since you need to run it at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        """
        self.net = torch.nn.Sequential(
          torch.nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
          torch.nn.BatchNorm2d(32),
          torch.nn.ReLU(),
          torch.nn.Conv2d(32, 64, kernel_size=7, stride=2, padding=3),
          torch.nn.BatchNorm2d(64),
          torch.nn.ReLU(),
          torch.nn.Conv2d(64, 128, kernel_size=7, stride=2, padding=3),
          torch.nn.BatchNorm2d(128),
          torch.nn.ReLU(),
          torch.nn.Conv2d(128, 256, kernel_size=7, stride=1, padding=3),
          torch.nn.BatchNorm2d(256),
          torch.nn.ReLU(),
          torch.nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
          torch.nn.BatchNorm2d(128),
          torch.nn.ReLU(),
          torch.nn.UpsamplingBilinear2d(scale_factor=2),
          torch.nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
          torch.nn.BatchNorm2d(64),
          torch.nn.ReLU(),
          torch.nn.UpsamplingBilinear2d(scale_factor=2),
          torch.nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
          torch.nn.BatchNorm2d(32),
          torch.nn.ReLU(),
          torch.nn.UpsamplingBilinear2d(scale_factor=2),
          torch.nn.Conv2d(32, 5, kernel_size=3, stride=1, padding=1),
          torch.nn.BatchNorm2d(5),
          torch.nn.ReLU()
        )


    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,6,H,W))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
              if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding strided
              convolution
        """
        _B, _N, H, W = x.shape
        z = self.net(x)[:,:,:H,:W]
        return z


model_factory = {
    'cnn': CNNClassifier,
    'fcn': FCN,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
