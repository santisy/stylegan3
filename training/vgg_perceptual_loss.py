import torch
from torch import nn
import torchvision


class VGGPerceptualLoss(nn.Module):
    DEFAULT_FEATURE_LAYERS = [0, 1, 2, 3]
    IMAGENET_RESIZE = (224, 224)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    IMAGENET_SHAPE = (1, 3, 1, 1)

    def __init__(self, resize=True, feature_layers=None, style_layers=None):
        super().__init__()
        self.resize = resize
        self.feature_layers = feature_layers or self.DEFAULT_FEATURE_LAYERS
        self.style_layers = style_layers or []
        features = torchvision.models.vgg16(pretrained=True).features
        self.blocks = nn.ModuleList([
            features[:4].eval(),
            features[4:9].eval(),
            features[9:16].eval(),
            features[16:23].eval(),
        ])
        for param in self.parameters():
            param.requires_grad = False
        self.register_buffer("mean", torch.tensor(self.IMAGENET_MEAN).view(self.IMAGENET_SHAPE))
        self.register_buffer("std", torch.tensor(self.IMAGENET_STD).view(self.IMAGENET_SHAPE))

    def _transform(self, tensor):
        #if tensor.shape != self.IMAGENET_SHAPE:
        #    tensor = tensor.repeat(self.IMAGENET_SHAPE)
        tensor = (tensor - self.mean) / self.std
        if self.resize:
            tensor = nn.functional.interpolate(tensor, mode='bilinear', size=self.IMAGENET_RESIZE, align_corners=False)
        return tensor

    def _calculate_gram(self, tensor):
        act = tensor.reshape(tensor.shape[0], tensor.shape[1], -1)
        return act @ act.permute(0, 2, 1)

    def forward(self, output, target):
        output = (output + 1) / 2.0
        target = (target + 1) / 2.0
        output, target = self._transform(output), self._transform(target)
        loss = 0.
        for i, block in enumerate(self.blocks):
            output, target = block(output), block(target)
            if i in self.feature_layers:
                loss += nn.functional.l1_loss(output, target)
            if i in self.style_layers:
                gram_output, gram_target = self._calculate_gram(output), self._calculate_gram(target)
                loss += nn.functional.l1_loss(gram_output, gram_target)
        return loss
    