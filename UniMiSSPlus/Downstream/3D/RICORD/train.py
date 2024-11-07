import torch
import numpy as np
import torch.backends.cudnn as cudnn
import os
from tqdm import tqdm
from nets.MiTPlus import model_plus
import matplotlib.pyplot as plt
from dataset.mydataset3D import Dataset3D, ValDataSet3D
from torch.utils import data
import argparse
from sklearn import metrics


parser = argparse.ArgumentParser()

parser.add_argument("-train_list", type=str, default='lists/RICORD_train.txt')
parser.add_argument("-val_list", type=str, default='lists/RICORD_val.txt')

parser.add_argument("-GPU", type=str, default='0')
parser.add_argument("-INPUT_SIZE", type=str, default='64, 128, 128')
parser.add_argument("-NUM_CLASSES", type=int, default=2, help='number of classes')
parser.add_argument("-BATCH_SIZE", type=int, default=8, help='number of batchsize')
parser.add_argument("-EPOCH", type=int, default=100, help='number of epoches')
parser.add_argument("-TRAIN_NUM", type=int, default=512, help='number of epoches')

parser.add_argument("-LEARNING_RATE", type=float, default=0.00005, help='number of learning rate')
parser.add_argument("-optimizer", type=str, default='AdamW', help='type of optimizer')

parser.add_argument("-deterministic", default=False, help="use this if you want to use deterministic")
parser.add_argument("-patience", type=int, default=30, help='number of patience')
parser.add_argument("-seed", type=int, default=123, help='number of seed')

parser.add_argument("-save_path", type=str, default='models/UniMiss_small', help='save path')
parser.add_argument("-pre_train", default=False, help="use this if you want to load pretrained weights")
parser.add_argument("-pre_train_path", type=str, default='models/UniMiss_small.pth', help='pretrained path')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=args.GPU
INPUT_SIZE = args.INPUT_SIZE
d, w, h = map(int, INPUT_SIZE.split(','))
NUM_CLASSES = args.NUM_CLASSES
BATCH_SIZE = args.BATCH_SIZE
EPOCH = args.EPOCH
TRAIN_NUM = args.TRAIN_NUM
STEPS = (TRAIN_NUM/BATCH_SIZE)*EPOCH

deterministic = args.deterministic

NAME = args.save_path

LEARNING_RATE = args.LEARNING_RATE
MOMENTUM = 0.9
POWER = 0.9
WEIGHT_DECAY = 5e-3   #3e-5


if deterministic:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(LEARNING_RATE, i_iter, STEPS, POWER)
    optimizer.param_groups[0]['lr'] = lr
    return lr

def cla_evaluate(label, binary_score, pro_score):
    acc = metrics.accuracy_score(label, binary_score)
    AP = metrics.average_precision_score(label, pro_score)
    auc = metrics.roc_auc_score(label, pro_score)
    CM = metrics.confusion_matrix(label, binary_score)
    sens = float(CM[1, 1]) / float(CM[1, 1] + CM[1, 0])
    spec = float(CM[0, 0]) / float(CM[0, 0] + CM[0, 1])
    return acc, auc, AP, sens, spec

def val_mode_cls(valloader, model):
    # valiadation
    pro_score = []
    label_val = []
    for index, batch in tqdm(enumerate(valloader)):
        data, label = batch
        data = data.cuda()

        model.eval()
        with torch.no_grad():
            pred = model(data)

        pro_score.append(torch.softmax(pred, dim=-1).cpu().data.numpy())
        label_val.append(label.data.numpy())

    pro_score = np.concatenate(pro_score, 0)
    binary_score = np.argmax(pro_score, axis=-1)
    label_val = np.concatenate(label_val, 0)

    val_acc, val_auc, val_AP, val_sens, val_spec = cla_evaluate(label_val, binary_score, pro_score[:, 1])

    return val_acc, val_auc, val_AP, val_sens, val_spec


def main():
    """Create the network and start the training."""

    cudnn.enabled = True

    ############# Create coarse segmentation network
    model = model_plus(norm_cfg3D='IN3', activation_cfg='LeakyReLU', img_size3D=[d, w, h], num_classes=NUM_CLASSES, pretrain=args.pre_train, pretrain_path=args.pre_train_path)

    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of Network Params: %.2f(e6)' % (total / 1e6))

    print(model)

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=0.99, nesterov=True)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    print(optimizer)

    model.cuda()

    model.train()
    model.float()

    loss_cls_CE = torch.nn.CrossEntropyLoss().cuda()

    cudnn.benchmark = True

    ############# Load training and validation data
    data_root = 'dataset/'
    data_train_list = args.train_list
    trainloader = data.DataLoader(Dataset3D(data_root, data_train_list, crop_size_3D=(d, w, h), max_iters=TRAIN_NUM),
                                  batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    data_val_list = args.val_list
    valloader = data.DataLoader(ValDataSet3D(data_root, data_val_list, crop_size_3D=(d, w, h)), batch_size=BATCH_SIZE, shuffle=False, num_workers=8,
                                pin_memory=True)

    path = NAME
    if not os.path.isdir(path):
        os.makedirs(path)
    f_path = os.path.join(path, 'outputxx.txt')

    val_score = []
    patience_counter = 0
    best_val_auc = 0

    ############# Start the training
    for epoch in range(EPOCH):

        train_loss_total = []

        for i_iter, batch in tqdm(enumerate(trainloader)):

            step = (TRAIN_NUM/BATCH_SIZE)*epoch+i_iter

            images, labels = batch
            images = images.cuda()
            labels = labels.cuda().long()

            optimizer.zero_grad()
            # lr = adjust_learning_rate(optimizer, step)
            model.train()
            preds = model(images)

            term = loss_cls_CE(preds, labels)

            term.backward()
            optimizer.step()

            train_loss_total.append(term.cpu().data.numpy())

        print("train_epoch%d: lossTotal=%f \n" % (epoch, np.nanmean(train_loss_total)))

        ############# Start the validation
        [val_acc, val_auc, val_AP, val_sens, val_spec] = val_mode_cls(valloader, model)
        line_val = "val%d: vacc=%f, vauc=%f, vAP=%f, vsens=%f, vspec=%f \n" % (epoch, val_acc, val_auc, val_AP, val_sens, val_spec)

        print(line_val)
        f = open(f_path, "a")
        f.write(line_val)

        ############# Plot val curve
        val_score.append(np.nanmean(val_auc))
        plt.figure()
        plt.plot(val_score, label='val auc score', color='blue', linestyle='--')
        plt.legend(loc='best')

        plt.savefig(os.path.join(path, 'jaccard.png'))
        plt.clf()
        plt.close()
        plt.show()

        plt.close('all')


        if np.nanmean(val_auc) > best_val_auc:
            torch.save(model.state_dict(), os.path.join(path, 'Best.pth'))
            best_epoch = epoch
            print("Epoch {:04d}: val_auc improved from {:.5f} to {:.5f}".format(best_epoch, best_val_auc, np.nanmean(val_auc)))
            best_val_auc = np.nanmean(val_auc)
            patience_counter = 0

        else: 
            print("Epoch {:04d}: val_auc did not improve from {:.5f} ".format(epoch, best_val_auc))
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early Stopping")
                break

        ############# Save final network
        torch.save(model.state_dict(), os.path.join(path, 'Final.pth'))


if __name__ == '__main__':
    main()
