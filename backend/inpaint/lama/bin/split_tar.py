#!/usr/bin/env python3


import tqdm
import webdataset as wds


def main(args):
    input_dataset = wds.Dataset(args.infile)
    output_dataset = wds.ShardWriter(args.outpattern)
    for rec in tqdm.tqdm(input_dataset):
        output_dataset.write(rec)


if __name__ == '__main__':
    import argparse

    aparser = argparse.ArgumentParser()
    aparser.add_argument('infile', type=str)
    aparser.add_argument('outpattern', type=str)

    main(aparser.parse_args())
