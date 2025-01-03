import glob
import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from test_util import test_all_case

from E_SegNet_3D import Swin3dENet
from utils import ramps, losses
from dataloaders.la_heart import LAHeart, RandomCrop, ToTensor

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/data/sjwlab/wuw/data/seg/pancreas_data(3d)/',
                    help='Name of Experiment')  # todo change dataset path
parser.add_argument('--pre_name', type=str, default='swin_tiny_patch244_window877_kinetics400_1k',
                    help='Name of Experiment')
parser.add_argument('--exp', type=str, default="pancreas1", help='model_name')  # todo model name
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')  # 6000
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=1, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.0001, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--ema_decay', type=float, default=0.999, help='ema_decay')
parser.add_argument('--consistency_type', type=str, default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "./outputs/" + args.exp + "/"
if os.path.exists(snapshot_path):
    na = args.exp[:-1]
    f = glob.glob("./outputs" + "/" + na + "*")
    if f:
        k = max([int((i.split("/")[-1]).replace(na, "")) for i in f])
    else:
        k = 0
    snapshot_path = "./model" + f"/{na}" + str(k + 1)
print(f"--save:{snapshot_path}")
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

num_classes = 2
patch_size = (64, 128, 128)
T = 0.1
Good_student = 0
with open("./dataset_pancreas/Pancreas" + '/Flods/test0.list',
          'r') as f:  # todo change test flod
    image_list = f.readlines()
image_list = [args.root_path + item.replace('\n', '') for item in image_list]


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)


def gateher_two_patch(vec):
    b, c, num = vec.shape
    cat_result = []
    for i in range(c - 1):
        temp_line = vec[:, i, :].unsqueeze(1)  # b 1 c
        star_index = i + 1
        rep_num = c - star_index
        repeat_line = temp_line.repeat(1, rep_num, 1)
        two_patch = vec[:, star_index:, :]
        temp_cat = torch.cat((repeat_line, two_patch), dim=2)
        cat_result.append(temp_cat)

    result = torch.cat(cat_result, dim=1)
    return result


if __name__ == "__main__":

    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    e_model = Swin3dENet(model_name=args.pre_name, image_size=patch_size,
                                    num_classes=num_classes).cuda()

    db_train = LAHeart(base_dir=train_data_path, split='train', train_flod='train0.list',
                       common_transform=transforms.Compose([RandomCrop(patch_size), ]),
                       sp_transform=transforms.Compose([ToTensor(), ]))

    trainloader = DataLoader(db_train, batch_sampler=None, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    optimizer = optim.AdamW(e_model.parameters(), eps=1e-8, betas=(0.9, 0.999),
                                lr=args.base_lr, weight_decay=0.00001)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))
    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr

    e_model.train()
    best_avg_metric = [0, 0, 0, 0]

    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            print('epoch:{}, i_batch:{}'.format(epoch_num, i_batch))
            volume_batch1, volume_label1 = sampled_batch[0]['image'], sampled_batch[0]['label']
            volume_batch2, volume_label2 = sampled_batch[1]['image'], sampled_batch[1]['label']
            # Transfer to GPU
            e_input, e_label = volume_batch1.cuda(), volume_label1.cuda()

            e_outputs = e_model(e_input)

            ## calculate the supervised loss
            e_loss_seg = F.cross_entropy(e_outputs[:labeled_bs], e_label[:labeled_bs])
            e_outputs_soft = F.softmax(e_outputs, dim=1)
            e_loss_seg_dice = losses.dice_loss(e_outputs_soft[:labeled_bs, 1, :, :, :], e_label[:labeled_bs] == 1)

            loss_total = e_loss_seg + e_loss_seg_dice
            # Network backward
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/total_loss', loss_total, iter_num)
            writer.add_scalar('loss/e_loss_seg', e_loss_seg, iter_num)
            writer.add_scalar('loss/e_loss_seg_dice', e_loss_seg_dice, iter_num)

            if iter_num % 50 == 0 and iter_num != 0:
                logging.info(
                    'iteration: %d Total loss : %f CE loss : %f Dice loss : %f' %
                    (iter_num, loss_total.item(), e_loss_seg.item(), e_loss_seg_dice.item(),))

            if iter_num % 500 == 0 and iter_num != 0:
                e_model.eval()
                avg_metric = test_all_case(e_model, e_model, image_list, num_classes=num_classes,
                                           patch_size=patch_size, stride_xy=16, stride_z=16,
                                           save_result=True, test_save_path=snapshot_path + f"/test/")
                if avg_metric[0] >= best_avg_metric[0]:
                    if best_avg_metric[0] != 0:
                        os.remove(glob.glob(os.path.join(snapshot_path, "*best*.pth"))[0])
                    best_avg_metric = avg_metric
                    save_mode_path_e_net = os.path.join(snapshot_path,
                                                          'best_iter_' + str(iter_num) + '.pth')
                    torch.save(e_model.state_dict(), save_mode_path_e_net)
                    logging.info("save model to {}...".format(save_mode_path_e_net))
                logging.info(f"best_avg_metric:{best_avg_metric}")
                e_model.train()

            ## change lr
            if iter_num % 12000 == 0 and iter_num != 0:
                lr_ = lr_ * 0.1
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            if iter_num >= max_iterations:
                break
            time1 = time.time()

            iter_num = iter_num + 1
            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            break

    save_mode_path_e_net = os.path.join(snapshot_path, 'E_SegNet_3D_iter_' + str(max_iterations) + '.pth')
    torch.save(e_model.state_dict(), save_mode_path_e_net)
    logging.info("save model to {}".format(save_mode_path_e_net))

    writer.close()
