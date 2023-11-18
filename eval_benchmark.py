import argparse
import os
import os.path as osp
from tqdm import tqdm
from tabulate import tabulate
from datetime import timedelta

# Import dv related package
import dv_processing as dv
import dv_toolkit as kit

# Import project config
from configs import Dataset, Denoisor

# Import evaluation metric
from python.src.utils.metric import EventStructuralRatio

if __name__ == '__main__':
    # Arguments settings
    parser = argparse.ArgumentParser(description='Run E-MLB benchmark.')
    parser.add_argument('-i', '--input_path',  type=str, default='./data', help='path to load dataset')
    parser.add_argument('-o', '--output_path', type=str, default='./results', help='path to output dataset')
    parser.add_argument('--denoisor', type=str, default='ynoise', help='choose a denoisor')
    parser.add_argument('--store_result', action='store_true', help='whether to store denoising result')
    parser.add_argument('--store_score', action='store_true', help='whether to store evaluation score')
    args = parser.parse_args()

    # Recursively load dataset
    datasets = Dataset(args.input_path)
    for i, dataset in enumerate(datasets):

        # Initialize on-screen info
        pbar = tqdm(dataset)
        table_header, table_data = ['Sequence', 'ESR Score'], list()

        for sequence in pbar:
            # Parse sequence info
            fpath, fclass, fname = sequence
            fname, fext = osp.splitext(fname)
            fdata = fpath.split('/')[-1]

            # Print progress bar info
            pbar.set_description(f"#Denoisor: {args.denoisor:>7s},  " +
                                 f"#Dataset: {fdata:>10s} ({i+1}/{len(datasets)}),  " +
                                 f"#Sequence: {fname:>10s}")

            # Load noisy file
            reader = kit.io.MonoCameraReader(f"{fpath}/{fclass}/{fname}{fext}")

            # Get Offline data
            data = reader.loadData()

            # Get resolutiong
            resolution = reader.getResolution("events")

            # Register event structural ratio
            metric = EventStructuralRatio(resolution)

            # Register denoisor
            model = Denoisor(args.denoisor, resolution)

            # Receive noise sequence
            model.accept(data["events"])

            # Perform event denoising
            data["events"] = model.generateEvents()

            # Store denoising result
            if args.store_result:
                output_path = f"{args.output_path}/{args.denoisor}/{fdata}/{fclass}"
                output_file = f"{output_path}/{fname}{fext}"
                if not osp.exists(output_path): os.makedirs(output_path)

                # Writing
                writer = kit.io.MonoCameraWriter(output_file, resolution)
                writer.writeData(data)

            # Store evaluation metric
            if args.store_score:
                score = metric.evalEventStorePerNumber(data["events"].toEventStore())
                table_data.append((fname, score.mean()))
        
        # Print ESR score
        if len(table_data) != 0:
            print(tabulate(table_data, headers=table_header, tablefmt='grid'))
