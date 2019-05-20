import argparse
import os

from sklearn.model_selection import train_test_split


def split_dataset(gt_path):

    print("Splitting...")
    root_path = os.path.dirname(gt_path)

    print("Fetching gt...")
    gt = get_gt(gt_path)

    # split the dataset
    print("Split gt...")
    train, val = train_test_split(gt)

    print("Writing files...")
    write_files(root_path, train, val)

    print("Finished splitting!")


def write_files(root_path, train, val):
    with open(os.path.join(root_path, 'train.txt'), 'w') as f:
        [f.write(' '.join(line)) for line in train]
    with open(os.path.join(root_path, 'dev.txt'), 'w') as f:
        [f.write(' '.join(line)) for line in val]


def get_gt(gt_path):
    # gt per line

    with open(gt_path, mode='r+') as file:
        # ground truth per line (tuple (filename, ints)) care, last int has a \n
        gt = [tuple(line.split(' ')) for line in file.readlines()]

    return gt


def create_folders(dataset_path):
    # create output folder
    root_path = os.path.join(os.path.split(dataset_path)[0], "split")
    output_images = os.path.join(root_path, "images")
    if not os.path.exists(output_images):
        os.makedirs(output_images)
    return output_images, root_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split dataset')

    parser.add_argument("--gt_file_path", required=True,
                        help="The path to the gt file")

    args = parser.parse_args()
    split_dataset(args.gt_file_path)
