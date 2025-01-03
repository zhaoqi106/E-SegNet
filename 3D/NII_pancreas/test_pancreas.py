import os
import argparse
import torch

from E_SegNet_3D import Swin3dENet
from test_util import test_all_case
from fvcore.nn import FlopCountAnalysis
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/data/sjwlab/wuw/data/seg/pancreas_data(3d)/', help='Name of Experiment')  # todo change dataset path
parser.add_argument('--pre_name', type=str, default='swin_tiny_patch244_window877_kinetics400_1k',
                    help='Name of Experiment')
parser.add_argument('--load_checkpoint', type=str,
                    default=r"outputs/pancreas1/best_iter_15500.pth", help='output channel of network')
parser.add_argument('--model', type=str,  default="pancreas1", help='model_name')                # todo change test model name
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
args = parser.parse_args()
torch.cuda.set_device(int(args.gpu))
# snapshot_path = "./prediction/"+args.model+'/'
test_save_path = "./prediction/"+args.model+'/' # change test save directory here
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

num_classes = 2
patch_size = (64, 128, 128)
with open("/data/sjwlab/wuw/project/seg/AgileFormer-main/SwinENet3d/dataset_pancreas/Pancreas" + '/Flods/test0.list', 'r') as f:                                         # todo change test flod
    image_list = f.readlines()
image_list = [args.root_path +item.replace('\n', '') for item in image_list] #+"/mri_norm2.h5"

def test_calculate_metric(epoch_num):
    model = Swin3dENet(model_name=args.pre_name,image_size=patch_size,num_classes=num_classes).cuda()
    save_mode_path = os.path.join(args.load_checkpoint)
    model.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))

    model.eval()
    avg_metric = test_all_case(model, model, image_list, num_classes=num_classes,
                               patch_size=patch_size, stride_xy=16, stride_z=16,
                               save_result=True, test_save_path=test_save_path)

    return avg_metric

if __name__ == '__main__':
    iters = 6000
    metric = test_calculate_metric(iters)
    print('iter:', iters)
    print(metric)
