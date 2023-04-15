from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-b', '--batch_size', default=10, type=int)
args = parser.parse_args()
print(args.batch_size)