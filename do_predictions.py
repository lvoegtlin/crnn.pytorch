import os

import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
from models.crnn import init_network
from datasets.datahelpers import default_loader
from utils.converter import LabelConverter

import argparse
import models

# provided models
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

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
        model.load_state_dict(checkpoint['state_dict'])
        converter = LabelConverter(alphabet, ignore_case=False)
        transform = transforms.Compose([
            transforms.Resize(30),
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

        return [str(n) for n in probs.numpy() if n != 0 and n != '\n'], preds


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Prediction")

    parser.add_argument("--alphabet", required=True, type=str,
                        help="path to the alphabet file")
    parser.add_argument("--model_path", required=True, type=str,
                        help="path to the model file (e.g. myBestModel.pth.tar)")
    parser.add_argument('--arch', default='densenet121', choices=model_names,
                        help='model architecture: {} (default: mobilenetv2_cifar)'.format(' | '.join(model_names)))
    parser.add_argument("--input_images", required=True, type=str,
                        help="Path to the input folder")
    parser.add_argument("--output_path", required=True, type=str,
                        help="Path to the output folder (does not have to exist)")
    args = parser.parse_args()

    # create outout folder
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    predictor = Prediction(alphabet_path=args.alphabet, model_path=args.model_path, model_arch=args.arch)

    # get all image files in output folder
    for root, _, files in os.walk(args.input_images):
        for file in files:
            if '.jpg' in file:
                _, prediction = predictor.predict(os.path.join(root, file))
                file_name, _ = os.path.splitext(file)
                # write file
                with open(os.path.join(args.output_path, file_name + '.txt'), mode='w') as f:
                    f.write(prediction)

    # Prediction(alphabet_path='./data/comp_alphabet.txt',
    #            model_path='./checkpoint/densenet121_adam_lr1.0e-03_wd5.0e-04_dataset_comp_set_alphabet_comp_alphabet_bsize32_height32_width20/densenet121_best.pth.tar',
    #            model_arch="densenet121").predict(input_image='./data/images/004708932_00056_l0.jpg')
