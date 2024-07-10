import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='Choose dataset preprocess type',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # dataloader
    parser.add_argument('--dataname', type=str, default='UI-PRMD', help='dataset name') # UI-PRMD or IRDS
    parser.add_argument('--input_type', type=str, default='features', help='type of data preprocessing') # raw, features, or tokens
    parser.add_argument('--downsample', type=int, default=5, help='downsampling rate') # 5 or 10
    parser.add_argument('--joints', type=int, nargs='+', default=[1, 8, 10, 12, 14, 17, 21], help='list of most relevant joints') # from 1 to 22 for kinect, 1 to 39 for vicon

    ## UI-PRMD options
    parser.add_argument('--device', type=str, default='kinect', help='device used to capture data') # kinect or vicon
    parser.add_argument('--correctness', type=str, default='correct', help='binary exercise quality assessment') # correct or incorrect
    parser.add_argument('--subdir', type=str, default='positions', help='subdirectory of data') # positions or angles or skeletons
    parser.add_argument('--m', type=int, default=1, help='motion number from 1-10')
    parser.add_argument('--s', type=int, default=1, help='subject number from 1-10')
    parser.add_argument('--e', type=int, default=1, help='episode number from 1-10')

    return parser.parse_args()