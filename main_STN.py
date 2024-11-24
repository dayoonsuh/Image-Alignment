"""Implements alignment algorithm."""

import argparse
from alignment_model import AlignmentModel
import time
import os


def main():
    """Main function to run the alignment model."""
    parser = argparse.ArgumentParser(description='CS59300CVD Assignment 1')
    parser.add_argument('-i', '--image_name', required=True,
                        type=str, help='Input image path')
    parser.add_argument('-m', '--metric', default='mse',
                        type=str, help='Metric to use for alignment')
    parser.add_argument('-lr', '--lr', default='0.001',
                        type=float, help='Learning rate')
    parser.add_argument('-s', '--steps', default='30',
                        type=int, help='Number of steps')
    args = parser.parse_args()
    print(args.image_name)

    if args.image_name == 'all':
        run_all(args)
        return

    image_name = 'data/part1/%s.jpg' % args.image_name
    model = AlignmentModel(image_name, metric=args.metric,
                           lr=args.lr, steps=args.steps)
    model.align()
    model.save('%s_%s_aligned.png' %
               (image_name.split('.')[0], args.metric))


def run_all(args):
    """Run the alignment model on all images."""

    for metric in ['mse', 'ncc']:
        for image_name in range(1, 7):
            image_name = 'data/part1/%d.jpg' % image_name
            # start_time = time.time()
            model = AlignmentModel(
                image_name, metric=metric, lr=args.lr, steps=args.steps)
            model.align()
            # end_time = time.time()
            # execution_time = end_time - start_time
            # print(f"{image_name} {metric} {execution_time}")
            model.save('%s_%s_aligned.png' %
                       (image_name.split('.')[0], metric))


if __name__ == '__main__':
    main()
