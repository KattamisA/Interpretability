import torch
from torch.autograd import Variable
from torchvision import models
import torch.nn as nn

model = getattr(models, "resnet18")(pretrained=False)


