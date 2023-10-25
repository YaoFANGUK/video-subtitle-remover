#!/usr/bin/env python3


import math
import os
import random

import braceexpand
import webdataset as wds

DEFAULT_CATS_FILE = os.path.join(os.path.dirname(__file__), '..', 'configs', 'places2-categories_157.txt')

def is_good_key(key, cats):
    return any(c in key for c in cats)


def main(args):
    if args.categories == 'nofilter':
        good_categories = None
    else:
        with open(args.categories, 'r') as f:
            good_categories = set(line.strip().split(' ')[0] for line in f if line.strip())

    all_input_files = list(braceexpand.braceexpand(args.infile))
    chunk_size = int(math.ceil(len(all_input_files) / args.n_read_streams))

    input_iterators = [iter(wds.Dataset(all_input_files[start : start + chunk_size]).shuffle(args.shuffle_buffer))
                       for start in range(0, len(all_input_files), chunk_size)]
    output_datasets = [wds.ShardWriter(args.outpattern.format(i)) for i in range(args.n_write_streams)]

    good_readers = list(range(len(input_iterators)))
    step_i = 0
    good_samples = 0
    bad_samples = 0
    while len(good_readers) > 0:
        if step_i % args.print_freq == 0:
            print(f'Iterations done {step_i}; readers alive {good_readers}; good samples {good_samples}; bad samples {bad_samples}')

        step_i += 1

        ri = random.choice(good_readers)
        try:
            sample = next(input_iterators[ri])
        except StopIteration:
            good_readers = list(set(good_readers) - {ri})
            continue

        if good_categories is not None and not is_good_key(sample['__key__'], good_categories):
            bad_samples += 1
            continue

        wi = random.randint(0, args.n_write_streams - 1)
        output_datasets[wi].write(sample)
        good_samples += 1


if __name__ == '__main__':
    import argparse

    aparser = argparse.ArgumentParser()
    aparser.add_argument('--categories', type=str, default=DEFAULT_CATS_FILE)
    aparser.add_argument('--shuffle-buffer', type=int, default=10000)
    aparser.add_argument('--n-read-streams', type=int, default=10)
    aparser.add_argument('--n-write-streams', type=int, default=10)
    aparser.add_argument('--print-freq', type=int, default=1000)
    aparser.add_argument('infile', type=str)
    aparser.add_argument('outpattern', type=str)

    main(aparser.parse_args())
