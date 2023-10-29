import argparse
import os.path as osp
from datetime import timedelta

# Import dv related package
import dv_processing as dv
import dv_toolkit as kit

# Import project config
from configs import Denoisor

# Import evaluation metric
from python.src.utils.metric import EventStructuralRatio


if __name__ == '__main__':
    # Arguments settings
    parser = argparse.ArgumentParser(description='Run single denoisor and evaluation.')
    parser.add_argument('-f', '--file', type=str, default='./data/demo/samples/demo-01.aedat4')
    parser.add_argument('--denoisor', type=str, default='ynoise', help='choose a denoisor')
    parser.add_argument('--parameters', type=lambda x: {k:int(v) for k,v in (i.split(':') for i in x.split(','))}, 
                        default="intThreshold: 1", help='set parameters of the denoisor')
    args = parser.parse_args()   

    # Load file to MonoCameraReader
    reader = kit.io.MonoCameraReader(args.file)
    
    # Get Offline data
    data = reader.loadData()

    # Register event structural ratio
    metric = EventStructuralRatio(reader.getResolution())

    # Print before filter
    score = metric.evalEventStorePerNumber(data["events"].toEventStore())
    print(f"Before filter >>>\n  {data['events']}, \n  ESR score: {score.mean():.3f}")

    # Initialize denoisor
    model = Denoisor(args.denoisor, reader.getResolution(), args.parameters)
    
    # Receive noise sequence
    model.accept(data["events"])
    
    # Perform event denoising
    data["events"] = model.generateEvents()

    # Print after filter
    score = metric.evalEventStorePerNumber(data["events"].toEventStore())
    print(f"After filter <<<\n  {data['events']}, \n  ESR score: {score.mean():.3f}")
