import torch
import torch.nn.functional as F

def extract_peak(heatmap, max_pool_ks=7, min_score=-5, max_det=100):
    """
       Your code here.
       Extract local maxima (peaks) in a 2d heatmap.
       @heatmap: H x W heatmap containing peaks (similar to your training heatmap)
       @max_pool_ks: Only return points that are larger than a max_pool_ks x max_pool_ks window around the point
       @min_score: Only return peaks greater than min_score
       @return: List of peaks [(score, cx, cy), ...], where cx, cy are the position of a peak and score is the
                heatmap value at the peak. Return no more than max_det peaks per image
    """
    # def isMax(heatmap, max_pool_ks, i, j):
    #     for r in range(max(0, i - (max_pool_ks // 2)), min(heatmap.size(0), i + max_pool_ks // 2)):
    #         for c in range(max(0, j - (max_pool_ks // 2)), min(heatmap.size(1), j + max_pool_ks // 2)):
    #             if heatmap[i, j] < heatmap[r, c]:
    #                 return False
        
    #     return True
    
    # peaks = []
    # for i in range(heatmap.size(0)):
    #     for j in range(heatmap.size(1)):
    #         if isMax(heatmap, max_pool_ks, i, j) and float(heatmap[i, j]) > min_score:
    #             peaks.append((heatmap[i, j], j, i))

    #         if len(peaks) == max_det:
    #             return peaks
            
    # return peaks
    # localMaxs = F.max_pool2d(heatmap[None, None], kernel_size=max_pool_ks, padding=max_pool_ks // 2, stride=1)

    # peaks = []
    # for i in range(localMaxs.size(2)):
    #     for j in range(localMaxs.size(3)):
    #         if localMaxs[0, 0, i, j] == heatmap[i, j] and heatmap[i, j] > min_score:
    #             peaks.append((heatmap[i, j], j, i))
    #             if len(peaks) == max_det:
    #                 return peaks

    # return peaks

    local_maxs, indices = F.max_pool2d(heatmap[None, None, :, :], kernel_size=max_pool_ks, padding=max_pool_ks // 2, stride=1, return_indices=True)

    is_peak = (heatmap >= local_maxs).float()
    
    mask = torch.logical_and(local_maxs > min_score, is_peak == 1.0)
    #mask = []
    x = torch.Tensor.bool(1,1,mask.size(2), mask.size(3))
    print(x.size())
    local_maxs = local_maxs[mask]
    indices = indices[mask]
    peaks, inds = torch.topk(local_maxs, min(max_det, len(local_maxs)))
    return [(peaks[i], indices[inds[i]] % heatmap.size(1), indices[inds[i]] // heatmap.size(1)) for i in range(peaks.size(0))]
    


class Detector(torch.nn.Module):
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
    
    class DownBlock(torch.nn.Module):
        def __init__(self, n_input, n_output, stride=1):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=3, padding=1, stride=stride, output_padding=1),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU()
            )
            
        def forward(self, x):
            return self.net(x)

    def __init__(self):
        """
           Your code here.
           Setup your detection network
        """
        super().__init__()

        c = 3
        l = []
        layers = [16, 32, 64, 128]
        for layer in layers:
            l.append(self.Block(c, layer, stride=2))
            c = layer

        layers = [128, 64, 32, 16]
        for layer in layers:
            l.append(self.DownBlock(c, layer, stride=2))
            c = layer
        
        self.feature_extractor = torch.nn.Sequential(*l)
        self.classifier = torch.nn.Conv2d(c, 3, 1)

    def forward(self, x):
        """
           Your code here.
           Implement a forward pass through the network, use forward for training,
           and detect for detection
        """
        x = self.feature_extractor(x)
        return self.classifier(x)

    def detect(self, image):
        """
           Your code here.
           Implement object detection here.
           @image: 3 x H x W image
           @return: Three list of detections [(score, cx, cy, w/2, h/2), ...], one per class,
                    return no more than 30 detections per image per class. You only need to predict width and height
                    for extra credit. If you do not predict an object size, return w=0, h=0.
           Hint: Use extract_peak here
           Hint: Make sure to return three python lists of tuples of (float, int, int, float, float) and not a pytorch
                 scalar. Otherwise pytorch might keep a computation graph in the background and your program will run
                 out of memory.
        """
        res = []
        hms = self(image[None, :, :, :])
        for i in range(hms.size(1)):
            peaks = []
            for s, cx, cy in extract_peak(hms[0, i], max_pool_ks=7, max_det=25):
                peaks.append((s, cx, cy, 0, 0))
            res.append(peaks)

        return res


def save_model(model):
    from torch import save
    from os import path
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'det.th'))


def load_model():
    from torch import load
    from os import path
    r = Detector()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'det.th'), map_location='cpu'))
    return r


if __name__ == '__main__':
    """
    Shows detections of your detector
    """
    from .utils import DetectionSuperTuxDataset
    dataset = DetectionSuperTuxDataset('dense_data/valid', min_size=0)
    import torchvision.transforms.functional as TF
    from pylab import show, subplots
    import matplotlib.patches as patches

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    fig, axs = subplots(3, 4)
    model = load_model().eval().to(device)
    for i, ax in enumerate(axs.flat):
        im, kart, bomb, pickup = dataset[i]
        ax.imshow(TF.to_pil_image(im), interpolation=None)
        for k in kart:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='r'))
        for k in bomb:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='g'))
        for k in pickup:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='b'))
        detections = model.detect(im.to(device))
        for c in range(3):
            for s, cx, cy, w, h in detections[c]:
                ax.add_patch(patches.Circle((cx, cy), radius=max(2 + s / 2, 0.1), color='rgb'[c]))
        ax.axis('off')
    show()