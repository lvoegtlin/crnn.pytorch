from utils.converter import LabelConverter
from tenacity import retry


@retry
def enrich_alphabet():
    alphabet_path = './data/alphabet_decode_5990_extend.txt'
    alphabet = ''

    with open(alphabet_path, mode='r', encoding='utf-8') as f:
        for line in f.readlines():
            alphabet += line.strip()

    check_alphabet(alphabet, alphabet_path)


def check_alphabet(alphabet, alphabet_path):
    with open('./data/comp_test/004708932_00055_l0.txt', 'r') as f:
        gt = f.readline()
        try:
            converter = LabelConverter(alphabet, ignore_case=False)
            converter.encode(gt)[0].numpy()
        except KeyError as e:
            # write char into alphabet file
            write_char_into_file(alphabet_path, e)
            raise RuntimeError


def write_char_into_file(alphabet_path, e):
    with open(alphabet_path, mode='a', encoding='utf-8') as f:
        f.write(e.args[0] + '\n')


if __name__ == '__main__':
    enrich_alphabet()
