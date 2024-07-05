import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Choose dataset preprocess type',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # dataloader
    parser.add_argument('--dataname', type=str, default='UI-PRMD', help='dataset name') # UI-PRMD or IRDS
    
    ## UI-PRMD options
    parser.add_argument('--device', type=str, default='kinect', help='device used to capture data') # kinect or vicon
    parser.add_argument('--correctness', type=str, default='correct', help='binary exercise quality assessment') # correct or incorrect
    parser.add_argument('--subdir', type=str, default='positions', help='subdirectory of data') # positions or angles or skeletons
    parser.add_argument('--m', type=int, default=1, help='motion number from 1-10')
    parser.add_argument('--s', type=int, default=1, help='subject number from 1-10')
    parser.add_argument('--e', type=int, default=1, help='episode number from 1-10')

    return parser.parse_args()