from toolbox import *
dict_hyper = {
    "v": 'origin',
    'e': 10,
    'b': 128,
    'lr': 0.01,
}
ls_loss = [
    [3.912035337, 0.020116397, 3.911873081, 0.02],
    [3.911809552, 0.020575025, 3.911561789, 0.04],
    [3.911499875, 0.021349949, 3.910597209, 0.024444444],
    [3.909137473, 0.03019041, 3.90747871, 0.035555556],
    [3.905344426, 0.032451923, 3.90274653, 0.035555556],
]

draw_loss_n_save(ls_loss, dict_hyper)
# from argparse import ArgumentParser

# parser = ArgumentParser()
# parser.add_argument('-b', '--batch_size', default=10, type=int)
# args = parser.parse_args()
# print(args.batch_size)
