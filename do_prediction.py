import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
from models.crnn import init_network
from datasets.datahelpers import default_loader
from utils.converter import LabelConverter

# load alphabet from file
alphabet = '0123456789'
# with open('./data/alphabet_decode_5990.txt', mode='r', encoding='utf-8') as f:
#     for line in f.readlines():
#         alphabet += line.strip()

img_name = './data/number_sequence/images/00001_19767937455632.jpg'
device = torch.device("cpu")
model_path = './checkpoint/densenet_cifar_rmsprop_lr1.0e-03_wd5.0e-04_bsize64_height32_width200/densenet_cifar_best.pth.tar'
# model_path = './checkpoint/model/densenet121_pretrained.pth'

model_params = {'architecture': "densenet_cifar",
                'num_classes': len(alphabet) + 1,
                'mean': (0.5,),
                'std': (0.5,)
                }
model = init_network(model_params)
model = model.to(device)

# load checkpoint
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['state_dict'])

converter = LabelConverter(alphabet, ignore_case=False)

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=model.meta['mean'], std=model.meta['std']),
])

print('image name: {}'.format(img_name))
img = default_loader(img_name)

plt.imshow(img)

# transform
img = transform(img)
img = img.unsqueeze(0)
img = img.to(device)

with torch.no_grad():
    log_probs = model(img)
    preds_strs = converter.best_path_decode(log_probs, strings=True).decode('utf-8')
    preds = converter.best_path_decode(log_probs, raw=True).decode('utf-8')

    print('pred: {}'.format(preds))
    print('pred strings: {}'.format(preds_strs))