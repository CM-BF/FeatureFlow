import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, help='path to Vimeo-90K dataset')
parser.add_argument('--out', type=str, required=True, help='path to output dataset place')
args = parser.parse_args()

def main():
    train_out_path = args.out + 'train'
    test_out_path = args.out + 'test'
    validation_out_path = args.out + 'validation'
    os.mkdir(train_out_path)
    os.mkdir(test_out_path)
    os.mkdir(validation_out_path)

    with open(args.dataset + '/tri_trainlist.txt', 'r') as f:
        train_paths = f.read().split('\n')
        test_paths = f.read().split('\n')
        print()

if __name__ == '__main__':
    main()
