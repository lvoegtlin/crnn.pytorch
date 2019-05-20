import argparse
import os
import shutil

from utils.converter import LabelConverter
from tenacity import retry
from tqdm import tqdm

from utils.split_dataset import split_dataset


@retry
def enrich_alphabet(file_path, alphabet, alphabet_path):

    check_alphabet(file_path, alphabet, alphabet_path)


def create_alphabet_string(alphabet_path):
    alphabet = ''
    with open(alphabet_path, mode='r', encoding='utf-8') as f:
        for line in f.readlines():
            alphabet += line.strip()
    return alphabet


def check_alphabet(file_path, alphabet, alphabet_path):
    with open(file_path, 'r') as f:
        gt = f.readline()
        try:
            converter = LabelConverter(alphabet, ignore_case=False)
            converter.encode(gt)[0].numpy()
        except KeyError as e:
            if not e.args[0]:
                return
            # write char into alphabet file
            if '\u3000' not in e.args[0] and ' ' not in e.args[0]:
                write_char_into_file(alphabet_path, e)
                raise RuntimeError


def write_char_into_file(alphabet_path, e):
    with open(alphabet_path, mode='a', encoding='utf-8') as f:
        f.write(e.args[0] + '\n')


def add_to_gt_file(file_path, converter, gt_path):
    # get file name from full file path
    # i dont need the file extension, because i already know that its .txt
    file_name, _ = os.path.splitext(os.path.basename(file_path))

    # read the line form the new file
    with open(file_path, mode='r') as file:
        line = file.readline()
        # encode the line
        encoded_line = []
        for char in line:
            try:
                encoded_line.append(converter.encode(char)[0].numpy()[0])
            except KeyError:
                encoded_line.append(' ')

    # write the line into the gt file
    with open(gt_path, mode='a', encoding='utf-8') as gt_file:
        gt_file.write(file_name + '.jpg' + ' ' + ' '.join([str(i)for i in encoded_line]) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_folder", required=True, type=str,
                        help="Path to the icdar comp 19 dataset")
    parser.add_argument("--alphabet", required=False, default=None, type=str,
                        help="Path to the alphabet file")
    parser.add_argument("--extend", required=False, action='store_true',
                        help="Should it extend the alphabet file with not found chars")
    parser.add_argument("--output_path", required=True, type=str,
                        help="The path to the output folder")

    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(os.path.join(args.output_path, "images"))

    gt_path = os.path.join(args.output_path, 'ground_truth.txt')
    # empties the existing gt file
    if os.path.exists(gt_path):
        open(gt_path, 'w').close()

    alphabet_string = create_alphabet_string(args.alphabet)

    converter = LabelConverter(alphabet_string, ignore_case=False)

    files_list = []
    for root, _, files in os.walk(args.dataset_folder):
        files_list.append(files)

    print("walking done. Found " + str(len(files)) + " files")

    for file in tqdm(files, ncols=150):
        file_path = os.path.join(root, file)
        # print("File nr: {} and name: {}".format(i, file))
        if ".txt" in file and "ground_truth.txt" not in file:
            if args.alphabet:
                enrich_alphabet(file_path, alphabet_string, args.alphabet)
                alphabet_string = create_alphabet_string(args.alphabet)
                converter = LabelConverter(alphabet_string, ignore_case=False)
            # write gt file
            add_to_gt_file(file_path, converter, gt_path)
        elif ".jpg" in file:
            shutil.copy(os.path.join(root, file), os.path.join(args.output_path, "images", file))

    split_dataset(gt_path)
