import torch
import numpy as np
import torch.backends.cudnn as cudnn
import os
from tqdm import tqdm
from nets.MiTPlus import model_plus
from dataset.mydataset3D import ValDataSet3D
from torch.utils import data
import argparse
from sklearn import metrics
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu
    torch.cuda.manual_seed_all(seed)  # all gpus

seed = np.random.randint(123)
set_seed(seed)

parser = argparse.ArgumentParser()


parser.add_argument("-test_list", type=str, default='lists/RICORD_test.txt')
parser.add_argument("-GPU", type=str, default='0')
parser.add_argument("-INPUT_SIZE", type=str, default='64, 128, 128')
parser.add_argument("-NUM_CLASSES", type=int, default=2, help='number of classes')
parser.add_argument("-BATCH_SIZE", type=int, default=8, help='number of batchsize')
parser.add_argument("-pre_train", default=False, help="use this if you want to load pretrained weights")
parser.add_argument("-pre_train_path", type=str, default=None, help='pretrained path')
parser.add_argument("-checkpoint_path", type=str, default='models/UniMiss_small/', help='checkpoint path')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=args.GPU
INPUT_SIZE = args.INPUT_SIZE
d, w, h = map(int, INPUT_SIZE.split(','))
NUM_CLASSES = args.NUM_CLASSES
NAME = args.checkpoint_path
BATCH_SIZE = args.BATCH_SIZE

def cla_evaluate(label, binary_score, pro_score):
    acc = metrics.accuracy_score(label, binary_score)
    AP = metrics.average_precision_score(label, pro_score)
    auc = metrics.roc_auc_score(label, pro_score)
    CM = metrics.confusion_matrix(label, binary_score)
    sens = float(CM[1, 1]) / float(CM[1, 1] + CM[1, 0])
    spec = float(CM[0, 0]) / float(CM[0, 0] + CM[0, 1])
    return acc, auc, AP, sens, spec

def test_mode_cls(valloader, model):
    # valiadation
    pro_score = []
    label_val = []
    for index, batch in tqdm(enumerate(valloader)):
        data, label = batch
        data = data.cuda()

        model.eval()
        with torch.no_grad():
            pred = model(data)
            pred = torch.softmax(pred, dim=-1)

        pro_score.append(pred.cpu().data.numpy())
        label_val.append(label.data.numpy())

    pro_score = np.concatenate(pro_score, 0)
    binary_score = np.argmax(pro_score, axis=-1)
    label_val = np.concatenate(label_val, 0)

    val_acc, val_auc, val_AP, val_sens, val_spec = cla_evaluate(label_val, binary_score, pro_score[:, 1])

    return val_acc, val_auc, val_AP, val_sens, val_spec


def main():
    """Create the network and start the training."""

    cudnn.enabled = True

    pre_train_path = args.pre_train_path

    ############# Create coarse segmentation network
    model = model_plus(norm_cfg3D='IN3', activation_cfg='LeakyReLU', img_size3D=[d, w, h], num_classes=NUM_CLASSES, pretrain=args.pre_train, pretrain_path=pre_train_path)

    print('*********loading from checkpoint: {}'.format(os.path.join(args.checkpoint_path, 'Final.pth')))
    model.load_state_dict(torch.load(os.path.join(args.checkpoint_path, 'Final.pth')))

    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of Network Params: %.2f(e6)' % (total / 1e6))

    model.cuda()
    model.float()

    cudnn.benchmark = True

    ############# Load testing data
    data_root = 'dataset/'
    data_test_list = args.test_list
    testloader = data.DataLoader(ValDataSet3D(data_root, data_test_list, crop_size_3D=(d, w, h)), batch_size=BATCH_SIZE, shuffle=False, num_workers=8,
                                pin_memory=True)

    path = NAME
    if not os.path.isdir(path):
        os.mkdir(path)
    f_path = os.path.join(path, 'outputxx.txt')

    ############# Start the testing
    [test_acc, test_auc, test_AP, test_sens, test_spec] = test_mode_cls(testloader, model)
    line_test = "test: tacc=%f, tauc=%f, tAP=%f, tsens=%f, tspec=%f \n" % (test_acc, test_auc, test_AP, test_sens, test_spec)

    print(line_test)
    f = open(f_path, "a")
    f.write(line_test)


if __name__ == '__main__':
    main()
