from utils.converter import LabelConverter

if __name__ == '__main__':
    alphabet_path = './data/alphabet_decode_5990.txt'
    alphabet = ''
    with open(alphabet_path, mode='r', encoding='utf-8') as f:
        for line in f.readlines():
            alphabet += line.strip()

    converter = LabelConverter(alphabet, ignore_case=False)

    with open('./data/comp_test/004708932_00055_l1.txt', 'r') as f:
        gt = f.readline()
        gt_converted = converter.encode(gt)
    print(gt_converted)
