"""
(Testing FPS)
Pixel Difference Networks for Efficient Edge Detection (accepted as an ICCV 2021 oral)
See paper in https://arxiv.org/abs/2108.07009

Author: Zhuo Su, Wenzhe Liu
Date: Aug 22, 2020
"""

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import argparse
import os
import platform
import time

from torch.utils.data import DataLoader
import torchvision
import torch
import skimage
from PIL import Image
from scipy import io as sio

import models
from utils import *
from edge_dataloader import BSDS_VOCLoader, BSDS_Loader, Multicue_Loader, NYUD_Loader, TestDataset
from models.convert_pidinet import convert_pidinet, convert_pidinet_test

IS_LINUX = True if platform.system()=="Linux" else False
dataset_base_dir = '/opt/dataset'if IS_LINUX else 'C:/Users/xavysp/dataset'

parser = argparse.ArgumentParser(description='PyTorch Diff Convolutional Networks (Train)')

parser.add_argument('--datadir', type=str, default=dataset_base_dir,
        help='dir to the dataset')
parser.add_argument('--dataset', type=str, default='BIPED',
        help='data settings for BSDS, Multicue and NYUD datasets')

parser.add_argument('--model', type=str, default='pidinet',
        help='model to train the dataset')
parser.add_argument('--sa', action='store_true', 
        help='use attention in diffnet')
parser.add_argument('--dil', action='store_true', 
        help='use dilation in diffnet')
parser.add_argument('--config', type=str, default='carv4',
        help='model configurations, please refer to models/config.py for possible configurations')
parser.add_argument('--seed', type=int, default=None, 
        help='random seed (default: None)')
parser.add_argument('--gpu', type=str, default='', 
        help='gpus available')

parser.add_argument('--epochs', type=int, default=150, 
        help='number of total epochs to run')
parser.add_argument('-j', '--workers', type=int, default=4, 
        help='number of data loading workers')
parser.add_argument('--eta', type=float, default=0.3, 
        help='threshold to determine the ground truth')
parser.add_argument('--checkpoint', type=str, default='checkpoint_015.pth.tar',
        help='checkpoint name')
parser.add_argument('--evaluate-converted', type=bool, default=True,
        help='convert the checkpoint to vanilla cnn, then evaluate')

args = parser.parse_args()

if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

def main():

    global args

    ### Refine args
    if args.seed is None:
        args.seed = int(time.time())
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.use_cuda = torch.cuda.is_available()
    device = torch.device('cpu' if torch.cuda.device_count() == 0
                          else 'cuda')

    dataset_setting_choices = ['BSDS', 'NYUD-image', 'NYUD-hha', 'Multicue-boundary-1', 
                'Multicue-boundary-2', 'Multicue-boundary-3', 'Multicue-edge-1',
                               'Multicue-edge-2', 'Multicue-edge-3', 'BIPED']
    if not isinstance(args.dataset, list): 
        assert args.dataset in dataset_setting_choices, 'unrecognized data setting %s, please choose from %s' % (str(args.dataset), str(dataset_setting_choices))
        args.dataset = list(args.dataset.strip().split('-')) 

    print(args)

    ### Create model
    model = getattr(models, args.model)(args)

    ## Load checkpoint
    checkpoint_dir = os.path.join('results','eval_results',args.checkpoint)
    checkpoint = torch.load(checkpoint_dir,
                                     map_location=device)
    if args.evaluate_converted:
        model.load_state_dict(convert_pidinet_test(checkpoint['state_dict'], args.config))
    else:
        model.load_state_dict(checkpoint['state_dict'])
    ### Transfer to cuda devices
    if args.use_cuda:
        model = torch.nn.DataParallel(model).cuda()
        print('cuda is used, with %d gpu devices' % torch.cuda.device_count())
    else:
        print('cuda is not used, the running might be slow')

    ### Load Data
    if 'BSDS' == args.dataset[0]:
        test_dataset = BSDS_VOCLoader(root=args.datadir, split="test", threshold=args.eta)
    elif 'Multicue' == args.dataset[0]:
        test_dataset = Multicue_Loader(root=args.datadir, split="test", threshold=args.eta, setting=args.dataset[1:])
    elif 'NYUD' == args.dataset[0]:
        test_dataset = NYUD_Loader(root=args.datadir, split="test", setting=args.dataset[1:])
    else:
        test_dataset = TestDataset(args.datadir, test_data='BIPED', img_width=1280, img_height=720,
                                   mean_bgr=[103.939, 116.779, 123.68], test_list='test_rgb.lst', arg=args)

    test_loader = DataLoader(
        test_dataset, batch_size=1, num_workers=args.workers, shuffle=False)

    test(test_loader, model, args)
    return


def test(test_loader, model, args):

    model.eval()

    end = time.time()
    res_img_dir = os.path.join('results','test',args.model+'_BIPED2'+args.dataset[0],'pred')
    res_mat_dir = os.path.join('results','test',args.model+'_BIPED2'+args.dataset[0],'mat')
    os.makedirs(res_img_dir, exist_ok=True)
    os.makedirs(res_mat_dir, exist_ok=True)
    for idx, (image, img_name) in enumerate(test_loader):

        with torch.no_grad():
            image = image.cuda() if args.use_cuda else image
            _, _, H, W = image.shape
            results = model(image)
            result = torch.squeeze(results[-1]).cpu().numpy()

        results_all = torch.zeros((len(results), 1, H, W))
        for i in range(len(results)):
            results_all[i, 0, :, :] = results[i]

        # torchvision.utils.save_image(1 - results_all,
        #                              os.path.join(res_img_dir, "%s.jpg" % img_name))
        sio.savemat(os.path.join(res_mat_dir, '%s.mat' % img_name), {'img': result})
        result = Image.fromarray((255-(result * 255)).astype(np.uint8))
        result.save(os.path.join(res_img_dir, "%s.png" % img_name))
        print('saved in> ',os.path.join(res_img_dir, "%s.png" % img_name))
    end = time.time() - end
    print('fps: %f' % (len(test_loader) / end))


if __name__ == '__main__':
    main()
    print('done')