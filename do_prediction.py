import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
from models.crnn import init_network
from datasets.datahelpers import default_loader
from utils.converter import LabelConverter


class Prediction:

    def __init__(self, alphabet_path, model_path, model_arch):
        alphabet = ''
        with open(alphabet_path, mode='r', encoding='utf-8') as f:
            for line in f.readlines():
                alphabet += line.strip()

        device = torch.device("cpu")
        model_params = {'architecture': model_arch,  # "mobilenetv2_cifar",
                        'num_classes': len(alphabet) + 1,
                        'mean': (0.5,),
                        'std': (0.5,)
                        }
        model = init_network(model_params)
        model = model.to(device)

        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        converter = LabelConverter(alphabet, ignore_case=False)
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=model.meta['mean'], std=model.meta['std']),
        ])

        self.alphabet = alphabet
        self.converter = converter
        self.model = model
        self.transform = transform
        self.device = device

    def predict(self, input_image):
        print('image name: {}'.format(input_image))
        img = default_loader(input_image)
        plt.imshow(img)
        # transform
        img = self.transform(img)
        img = img.unsqueeze(0)
        img = img.to(self.device)

        with torch.no_grad():
            log_probs = self.model(img)
            preds_strs, _ = self.converter.best_path_decode(log_probs, strings=True)
            preds_strs = preds_strs.decode('utf-8')
            preds, probs = self.converter.best_path_decode(log_probs, raw=True)
            preds = preds.decode('utf-8')

            print('pred: {}'.format(preds))
            print('pred strings: {}'.format(preds_strs))

        return [str(n) for n in probs.numpy() if n != 0 and n != '\n']


if __name__ == '__main__':
    Prediction(alphabet_path='./data/alphabet_decode_5990.txt',
               model_path='./checkpoint/pre_trained/mobilenetv2_cifar_pretrained.pth',
               model_arch="mobilenetv2_cifar").predict(input_image='./data/images/00000002.jpg')
