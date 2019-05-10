import argparse
import os
import numpy as np

from do_prediction import Prediction

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="evaluate helper")
    parser.add_argument("--input_path", required=True,
                        help="path to the input folder (contains images folder and the ground_truth.txt file)")
    parser.add_argument("--alphabet", required=True,
                        help="path to the alphabet")
    parser.add_argument("--model_path", required=True,
                        help="path to the pre-trained model")
    parser.add_argument("--model_arch", required=True,
                        help="architecture of the provided model (e.g. mobilenetv2_cifar)")

    args = parser.parse_args()

    prediction = Prediction(args.alphabet, args.model_path, args.model_arch)

    amount_of_chars = 0
    TP = 0
    with open(os.path.join(args.input_path, "ground_truth.txt")) as f:
        for line in f.readlines():
            line_array = line.split(' ')
            file_name = line_array[0]
            gt = np.asarray([int(i) for i in line_array[1:]])
            pred = [int(i) for i in prediction.predict(os.path.join(args.input_path, "images", file_name))]
            amount_of_chars += len(gt)
            TP += np.intersect1d(pred, gt).size

    print("TP: " + str(TP))
    print("amount of chars: " + str(amount_of_chars))
    print("acc: " + str(TP/amount_of_chars))
