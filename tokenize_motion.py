import torch
import numpy as np

import models.vqvae as vqvae
import options.option_vq as option_vq
from dataset import dataset_VQ
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    # --- Network
    args = option_vq.get_args_parser()

    args.dataname = 't2m' 
    #args.nb_joints = 22
    args.resume_pth = 'pretrained/VQVAE/net_last.pth'
    args.resume_trans = 'pretrained/VQTransformer_corruption05/net_best_fid.pth'
    args.down_t = 2
    args.depth = 3
    args.block_size = 51
    net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                        args.nb_code,
                        args.code_dim,
                        args.output_emb_width,
                        args.down_t,
                        args.stride_t,
                        args.width,
                        args.depth,
                        args.dilation_growth_rate)

    # --- Load Checkpoints
    print ('loading checkpoint from {}'.format(args.resume_pth))
    ckpt = torch.load(args.resume_pth, map_location='cpu')
    net.load_state_dict(ckpt['net'], strict=True)
    net.eval()
    net.cuda()

    mean = torch.from_numpy(np.load('./checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy')).cuda()
    std = torch.from_numpy(np.load('./checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy')).cuda()
    print("---checkpoints loaded")

    # --- Dataloader
    train_loader = dataset_VQ.DATALoader(args.dataname,
                                            args.batch_size,
                                            window_size=args.window_size,
                                            unit_length=2**args.down_t)
    train_loader_iter = dataset_VQ.cycle(train_loader)

    # --- Get Motion
    gt_motion = next(train_loader_iter)
    print("---gt_motion: ", gt_motion.shape, "\n", gt_motion)
    gt_motion = gt_motion.cuda().float() # (bs, 64, dim) = (128, 64, 263)

    # --- Get Token
    encoded = net.encode(gt_motion)
    print("---encoded: ", encoded.shape, "\n", encoded)
    np.save('encoded.npy', encoded.detach().cpu().numpy())
    print("---encoded.npy saved")

    # --- Decode Tokens
    decoded = net.forward_decoder(encoded)
    decoded = decoded.view(128, 64, 263)
    print("---decoded: ", decoded.shape, "\n", decoded)
    # np.save('decoded.npy', encoded.detach().cpu().numpy())
    # print("---decoded.npy saved")

    # if decoded == gt_motion:
    #     print("decoded motion is equal to input ground truth motion")
   

    