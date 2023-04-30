import os
import time
import os.path as osp
import argparse
from tqdm import tqdm

from evtool.dvs import DvsFile
from configs import Dataset, Denoisor


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deployment of EMLB benchmark')
    parser.add_argument('-i', '--input_path',  type=str, default='./data', help='path to load dataset')
    parser.add_argument('-o', '--output_path', type=str, default='./results', help='path to output dataset')
    parser.add_argument('-d', '--denoisor',    type=str, default="mlpf", help='choose denoisors')
    parser.add_argument('--excl_hotpixel',     type=int, default=-1)
    parser.add_argument('--output_file_type',  type=str, default='pkl', help='output file type')

    args = parser.parse_args()

    model, datasets = Denoisor(args.denoisor), Dataset(args.input_path)
    for i, dataset in enumerate(datasets):
        pbar = tqdm(dataset)
        for sequence in pbar:
            fpath, fclass, fname = sequence
            fname, fext = osp.splitext(fname)
            fdata = fpath.split('/')[-1]
            
            # Print progress bar info
            pbar.set_description(f"{args.denoisor:>7s}"
                                 f"{fdata:>10s} ({i+1}/{len(datasets)})"
                                 f"{fname:>10s}")

            # Load noisy file
            data = DvsFile.load(osp.join(fpath, fclass, fname + fext))
            if args.excl_hotpixel > 0:
                idx = data['events'].hotpixel(data['size'], thres=1000)
                data['events'] = data['events'][idx]

            # Start inference
            data = model.run(data)

            # Save inference result
            if args.output_path is not None:
                output_file = f"{args.output_path}/{args.denoisor}/{fclass}/{fdata}/{fname}.{args.output_file_type}"
                output_dir, _ = osp.split(output_file)
                if not osp.exists(output_dir): os.makedirs(output_dir)
                DvsFile.save(data, output_file)
