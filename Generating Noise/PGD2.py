# Requires PyTorch (through env)

from collections import OrderedDict
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

########### VARIABLES (CHANGE as needed) #################
image_path = 'logos/yt.png'                                       # LOAD Clean Image
save_location = 'dataset/target_brand/youtube/logo_perturb.png'   # SAVE noisy image
correct_label = 239 # Change to label corresponding to logo
                    # 239 = YouTube
                    # 83 = Instagram
                    # 112 = Facebook
                    # (See brand277_target.json for more)

epsilon = 0.02  # Maximum perturbation- RAISE THIS to raise noise
step_size = 0.007  # Size of each PGD step- RAISE THIS to raise noise (keep under epsilon)
num_steps = 50  # Number of PGD steps- RAISE THIS to raise success of noise, will increase running time
######################################


############ MODEL CODE (Given by Fujiao Ji) ##########################
#Load the pre-trained model 
class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-10)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                                        self.dilation, self.groups)
def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                                     padding=0, bias=bias)


def tf2th(conv_weights):
    """Possibly convert HWIO to OIHW."""
    if conv_weights.ndim == 4:
        conv_weights = conv_weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(conv_weights)


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.

    Follows the implementation of "Identity Mappings in Deep Residual Networks":
    https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua

    Except it puts the stride on 3x3 conv when available.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout//4

        self.gn1 = nn.GroupNorm(32, cin)
        self.conv1 = conv1x1(cin, cmid)
        self.gn2 = nn.GroupNorm(32, cmid)
        self.conv2 = conv3x3(cmid, cmid, stride)    # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cmid)
        self.conv3 = conv1x1(cmid, cout)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride)

    def forward(self, x):
        out = self.relu(self.gn1(x))

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(out)

        # Unit's branch
        out = self.conv1(out)
        out = self.conv2(self.relu(self.gn2(out)))
        out = self.conv3(self.relu(self.gn3(out)))

        return out + residual

    def load_from(self, weights, prefix=''):
        convname = 'standardized_conv2d'
        with torch.no_grad():
            self.conv1.weight.copy_(tf2th(weights[f'{prefix}a/{convname}/kernel']))
            self.conv2.weight.copy_(tf2th(weights[f'{prefix}b/{convname}/kernel']))
            self.conv3.weight.copy_(tf2th(weights[f'{prefix}c/{convname}/kernel']))
            self.gn1.weight.copy_(tf2th(weights[f'{prefix}a/group_norm/gamma']))
            self.gn2.weight.copy_(tf2th(weights[f'{prefix}b/group_norm/gamma']))
            self.gn3.weight.copy_(tf2th(weights[f'{prefix}c/group_norm/gamma']))
            self.gn1.bias.copy_(tf2th(weights[f'{prefix}a/group_norm/beta']))
            self.gn2.bias.copy_(tf2th(weights[f'{prefix}b/group_norm/beta']))
            self.gn3.bias.copy_(tf2th(weights[f'{prefix}c/group_norm/beta']))
            if hasattr(self, 'downsample'):
                w = weights[f'{prefix}a/proj/{convname}/kernel']
                self.downsample.weight.copy_(tf2th(w))


class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor, head_size=21843, zero_head=False):
        super().__init__()
        wf = width_factor    # shortcut 'cause we'll use it a lot.

        # The following will be unreadable if we split lines.
        # pylint: disable=line-too-long
        self.root = nn.Sequential(OrderedDict([
                ('conv', StdConv2d(3, 64*wf, kernel_size=7, stride=2, padding=3, bias=False)),
                ('pad', nn.ConstantPad2d(1, 0)),
                ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0)),
                # The following is subtly not the same!
                # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        self.body = nn.Sequential(OrderedDict([
                ('block1', nn.Sequential(OrderedDict(
                        [('unit01', PreActBottleneck(cin=64*wf, cout=256*wf, cmid=64*wf))] +
                        [(f'unit{i:02d}', PreActBottleneck(cin=256*wf, cout=256*wf, cmid=64*wf)) for i in range(2, block_units[0] + 1)],
                ))),
                ('block2', nn.Sequential(OrderedDict(
                        [('unit01', PreActBottleneck(cin=256*wf, cout=512*wf, cmid=128*wf, stride=2))] +
                        [(f'unit{i:02d}', PreActBottleneck(cin=512*wf, cout=512*wf, cmid=128*wf)) for i in range(2, block_units[1] + 1)],
                ))),
                ('block3', nn.Sequential(OrderedDict(
                        [('unit01', PreActBottleneck(cin=512*wf, cout=1024*wf, cmid=256*wf, stride=2))] +
                        [(f'unit{i:02d}', PreActBottleneck(cin=1024*wf, cout=1024*wf, cmid=256*wf)) for i in range(2, block_units[2] + 1)],
                ))),
                ('block4', nn.Sequential(OrderedDict(
                        [('unit01', PreActBottleneck(cin=1024*wf, cout=2048*wf, cmid=512*wf, stride=2))] +
                        [(f'unit{i:02d}', PreActBottleneck(cin=2048*wf, cout=2048*wf, cmid=512*wf)) for i in range(2, block_units[3] + 1)],
                ))),
        ]))
        # pylint: enable=line-too-long

        self.zero_head = zero_head
        self.head = nn.Sequential(OrderedDict([
                ('gn', nn.GroupNorm(32, 2048*wf)),
                ('relu', nn.ReLU(inplace=True)),
                ('avg', nn.AdaptiveAvgPool2d(output_size=1)),
                ('conv', nn.Conv2d(2048*wf, head_size, kernel_size=1, bias=True)),
        ]))

    def features(self, x):
        x = self.head[:-1](self.body(self.root(x)))

        return x.squeeze(-1).squeeze(-1)

    def forward(self, x):
        x = self.head(self.body(self.root(x)))
        assert x.shape[-2:] == (1, 1)    # We should have no spatial shape left.
        return x[...,0,0]

    def load_from(self, weights, prefix='resnet/'):
        with torch.no_grad():
            self.root.conv.weight.copy_(tf2th(weights[f'{prefix}root_block/standardized_conv2d/kernel']))    # pylint: disable=line-too-long
            self.head.gn.weight.copy_(tf2th(weights[f'{prefix}group_norm/gamma']))
            self.head.gn.bias.copy_(tf2th(weights[f'{prefix}group_norm/beta']))
            if self.zero_head:
                nn.init.zeros_(self.head.conv.weight)
                nn.init.zeros_(self.head.conv.bias)
            else:
                self.head.conv.weight.copy_(tf2th(weights[f'{prefix}head/conv2d/kernel']))    # pylint: disable=line-too-long
                self.head.conv.bias.copy_(tf2th(weights[f'{prefix}head/conv2d/bias']))

            for bname, block in self.body.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, prefix=f'{prefix}{bname}/{uname}/')

KNOWN_MODELS = OrderedDict([
        ('BiT-M-R50x1', lambda *a, **kw: ResNetV2([3, 4, 6, 3], 1, *a, **kw)),
        ('BiT-M-R50x3', lambda *a, **kw: ResNetV2([3, 4, 6, 3], 3, *a, **kw)),
        ('BiT-M-R101x1', lambda *a, **kw: ResNetV2([3, 4, 23, 3], 1, *a, **kw)),
        ('BiT-M-R101x3', lambda *a, **kw: ResNetV2([3, 4, 23, 3], 3, *a, **kw)),
        ('BiT-M-R152x2', lambda *a, **kw: ResNetV2([3, 8, 36, 3], 2, *a, **kw)),
        ('BiT-M-R152x4', lambda *a, **kw: ResNetV2([3, 8, 36, 3], 4, *a, **kw)),
        ('BiT-S-R50x1', lambda *a, **kw: ResNetV2([3, 4, 6, 3], 1, *a, **kw)),
        ('BiT-S-R50x3', lambda *a, **kw: ResNetV2([3, 4, 6, 3], 3, *a, **kw)),
        ('BiT-S-R101x1', lambda *a, **kw: ResNetV2([3, 4, 23, 3], 1, *a, **kw)),
        ('BiT-S-R101x3', lambda *a, **kw: ResNetV2([3, 4, 23, 3], 3, *a, **kw)),
        ('BiT-S-R152x2', lambda *a, **kw: ResNetV2([3, 8, 36, 3], 2, *a, **kw)),
        ('BiT-S-R152x4', lambda *a, **kw: ResNetV2([3, 8, 36, 3], 4, *a, **kw)),
])
def pad_to_square(image):
    width, height = image.size
    max_wh = max(width, height)
    p_left = (max_wh - width) // 2
    p_top = (max_wh - height) // 2
    p_right = max_wh - width - p_left
    p_bottom = max_wh - height - p_top
    padding = (p_left, p_top, p_right, p_bottom)
    return transforms.functional.pad(image, padding, fill=255)

if __name__ == '__main__':
    """Load the classifier"""
    # The model can classify the input logo to 277 classes
    siamese_chkpt = "models/resnetv2_rgb_new.pth.tar"  # path for the saved model
    device = 'cpu'  # or 'cuda' if you're using a GPU
    model = KNOWN_MODELS["BiT-M-R50x1"](head_size=277, zero_head=True)

    # Load weights
    weights = torch.load(siamese_chkpt, map_location=device)
    weights = weights['model'] if 'model' in weights.keys() else weights
    new_state_dict = OrderedDict()

    for k, v in weights.items():
        if 'module.' in k:
            name = k.split('module.')[1]
        else:
            name = k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    model.to(device)
    model.eval()





####################### MY CODE TO GENERATE NOISE ########################
image = Image.open(image_path)
if image.mode != 'RGB':
    image = image.convert('RGB')  # Convert grayscale to RGB if needed


 
transform = transforms.Compose([
    transforms.Lambda(pad_to_square),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
])
image = transform(image).unsqueeze(0)  # Add batch dimension

# Define the true label
true_label = torch.tensor(correct_label, dtype=torch.long)  # Replace with the correct label for your image

# PGD: Generate adversarial example using iterative steps
perturbed_image = image.clone().detach()
perturbed_image.requires_grad = True

for _ in range(num_steps):
    perturbed_image.requires_grad = True

    # Forward pass
    logits = model(perturbed_image)
    loss = F.cross_entropy(logits, true_label.unsqueeze(0))

    # Backward pass
    model.zero_grad()
    loss.backward()

    # Add a small step in the direction of the gradient
    with torch.no_grad():
        perturbed_image = perturbed_image + step_size * perturbed_image.grad.sign()
        
        # Project back into the epsilon-ball and clip to [0,1]
        perturbed_image = torch.max(torch.min(perturbed_image, image + epsilon), image - epsilon)
        perturbed_image = torch.clamp(perturbed_image, 0, 1)

    # Zero gradients for next step
    perturbed_image.grad = None

# Test adversarial image
with torch.no_grad():
    logits = model(perturbed_image)
    v_possibility = F.softmax(logits, dim=-1)
    v_pred = torch.argmax(v_possibility, dim=-1)
    max_possibility = torch.gather(v_possibility, dim=-1, index=v_pred.unsqueeze(-1)).squeeze(-1)

print("Adversarial Prediction:", v_pred.item())
print("True Label:", true_label)
print("Adversarial Confidence:", max_possibility.item())

# Save perturbed image
perturbed_image_pil = transforms.ToPILImage()(perturbed_image.squeeze())
perturbed_image_pil.save(save_location)
