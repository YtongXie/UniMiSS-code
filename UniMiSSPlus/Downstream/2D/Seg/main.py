import torch
import numpy as np
import torch.backends.cudnn as cudnn
import os
from tqdm import tqdm
from net.MiTPlus import MiTPlus
from dataset.my_datasets import MyDataSet_seg, MyValDataSet_seg
from torch.utils import data
import argparse
from scipy import ndimage


parser = argparse.ArgumentParser()
parser.add_argument("-train_list", type=str, default='dataset/Training_seg_0.2.txt')
parser.add_argument("-test_list", type=str, default='dataset/Test_seg.txt')

parser.add_argument("-GPU", type=str, default='0')
parser.add_argument("-INPUT_SIZE", type=str, default='224, 224')
parser.add_argument("-NUM_CLASSES", type=int, default=3, help='number of epoches')
parser.add_argument("-BATCH_SIZE", type=int, default=16, help='number of epoches')
parser.add_argument("-EPOCH", type=int, default=100, help='number of epoches')
parser.add_argument("-TRAIN_NUM", type=int, default=1691, help='number of epoches')
parser.add_argument("-LEARNING_RATE", type=float, default=0.0001, help='number of learning rate')
parser.add_argument("-optimizer", type=str, default='AdamW', help='type off optimizer')

parser.add_argument("-deterministic", default=True, help="use this if you want to use deterministic")

parser.add_argument("-save_path", type=str, default='xxx/', help='save path')
parser.add_argument("-pre_train", default=False, help="use this if you want to load pretrained weights")
parser.add_argument("-pre_train_path", type=str, default='UniMiss.pth', help='pretrained path')

parser.add_argument("-num_works", type=int, default=8, help="num of workers")
parser.add_argument("-seed", type=int, default=234, help='number of seed')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=args.GPU
INPUT_SIZE = args.INPUT_SIZE
w, h = map(int, INPUT_SIZE.split(','))
NUM_CLASSES = args.NUM_CLASSES
BATCH_SIZE = args.BATCH_SIZE
EPOCH = args.EPOCH
TRAIN_NUM = args.TRAIN_NUM
STEPS = (TRAIN_NUM/BATCH_SIZE)*EPOCH

deterministic = args.deterministic

NAME = args.save_path
pre_train = args.pre_train
pre_train_path = args.pre_train_path

LEARNING_RATE = args.LEARNING_RATE
MOMENTUM = 0.9
POWER = 0.9
WEIGHT_DECAY = 0.0005

NUM_WORK = args.num_works

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


def val_mode_seg(valloader, model, path, epoch):
    dice_c = []
    dice_h = []
    dice_l = []
    dice = []

    # sen = []
    # spe = []
    # acc = []
    jac_c= []
    jac_h = []
    jac_l = []
    jac = []

    for index, batch in enumerate(valloader):

        data, mask, name = batch
        data = data.cuda()
        mask = mask.data.numpy()
        val_mask = np.int64(mask > 0)
        # print(data.shape)

        model.eval()
        with torch.no_grad():
            pred = model(data)

        pred = torch.sigmoid(pred).cpu().data.numpy()
        pred_binary = np.int64(pred>=0.5)

        # metric for clavicle
        y_true_f = val_mask[:, 0].reshape(val_mask.shape[-1]*val_mask.shape[-2], order='F')
        y_pred_f = pred_binary[:, 0].reshape(pred_binary.shape[-1]*pred_binary.shape[-2], order='F')
        intersection = np.float(np.sum(y_true_f * y_pred_f))
        dice_c.append((2. * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f) + 0.01))
        jac_c.append(intersection / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection + 0.01))

        # metric for heart
        y_true_f = val_mask[:, 1].reshape(val_mask.shape[-1]*val_mask.shape[-2], order='F')
        y_pred_f = pred_binary[:, 1].reshape(pred_binary.shape[-1]*pred_binary.shape[-2], order='F')
        intersection = np.float(np.sum(y_true_f * y_pred_f))
        dice_h.append((2. * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f) + 0.01))
        jac_h.append(intersection / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection + 0.01))

        # metric for lung
        y_true_f = val_mask[:, 2].reshape(val_mask.shape[-1]*val_mask.shape[-2], order='F')
        y_pred_f = pred_binary[:, 2].reshape(pred_binary.shape[-1]*pred_binary.shape[-2], order='F')
        intersection = np.float(np.sum(y_true_f * y_pred_f))
        dice_l.append((2. * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f) + 0.01))
        jac_l.append(intersection / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection + 0.01))


        # metric for all
        y_true_f = val_mask.reshape(val_mask.shape[0]*val_mask.shape[1]*val_mask.shape[2]*val_mask.shape[3], order='F')
        y_pred_f = pred_binary.reshape(pred_binary.shape[0]*pred_binary.shape[1]*pred_binary.shape[2]*val_mask.shape[3], order='F')
        intersection = np.float(np.sum(y_true_f * y_pred_f))
        dice.append((2. * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f) + 0.01))
        jac.append(intersection / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection + 0.01))

    return np.array(dice), np.array(jac), np.array(dice_c), np.array(jac_c), np.array(dice_h), np.array(jac_h), np.array(dice_l), np.array(jac_l)


def Jaccard(pred, mask):
    pred = torch.sigmoid(pred).cpu().data.numpy()
    pred_binary = np.int64(pred>=0.5)

    mask = mask.cpu().data.numpy()

    y_true_f = mask.reshape(mask.shape[0] * mask.shape[1] * mask.shape[2] * mask.shape[3], order='F')
    y_pred_f = pred_binary.reshape(pred_binary.shape[0] * pred_binary.shape[1] * pred_binary.shape[2] * pred_binary.shape[3], order='F')

    intersection = np.float(np.sum(y_true_f * y_pred_f))
    jac_score = intersection / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection)

    return jac_score


def dice_loss(predict, target):

	smooth = 1e-5

	y_true_f = target.contiguous().view(target.shape[0], -1)
	y_pred_f = predict.contiguous().view(predict.shape[0], -1)
	intersection = torch.sum(torch.mul(y_pred_f, y_true_f), dim=1)
	union = torch.sum(y_pred_f, dim=1) + torch.sum(y_true_f, dim=1) + smooth
	dice_score = (2.0 * intersection / union)

	dice_loss = 1 - dice_score

	return dice_loss


def Fuse_Dice_CE(predicts, targets):
    BCE_loss=torch.nn.BCEWithLogitsLoss()
    loss_BCE = BCE_loss(predicts, targets)

    preds = torch.tensor(torch.sigmoid(predicts) >= 0.5, dtype=torch.long)
    dice_loss0 = dice_loss(preds[:, 0, :, :], targets[:, 0, :, :])
    dice_loss1 = dice_loss(preds[:, 1, :, :], targets[:, 1, :, :])
    dice_loss2 = dice_loss(preds[:, 2, :, :], targets[:, 2, :, :])

    loss_D = (dice_loss0.mean() + dice_loss1.mean() + dice_loss2.mean())/3.0

    return loss_D, loss_BCE



def main():
    """Create the network and start the training."""
    cudnn.enabled = True

    ############# Create coarse segmentation network
    model = MiTPlus(norm_cfg2D='IN2', activation_cfg='LeakyReLU', is_proj1=True, img_size2D=224, num_classes=NUM_CLASSES)

    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of Network Params: %.2f(e6)' % (total / 1e6))

    # print(model)

    if pre_train:
        print('*********loading from checkpoint ssl: {}'.format(pre_train_path))
        print('*********loading from checkpoint ssl --> epoch: {}'.format(str(torch.load(pre_train_path, map_location="cpu")['epoch'])))

        pre_type = 'student'  #teacher student  
        if pre_type == 'teacher': 
            pre_dict_ori = torch.load(pre_train_path, map_location="cpu")[pre_type]
            pre_dict_ori = {k.replace("backbone.", ""): v for k, v in pre_dict_ori.items()}
            print('Teacher: length of pre-trained layers: %.f' % (len(pre_dict_ori)))
        elif pre_type == 'student': 
            pre_dict_ori = torch.load(pre_train_path, map_location="cpu")[pre_type]
            pre_dict_ori = {k.replace("module.backbone.", ""): v for k, v in pre_dict_ori.items()}
            print('Student: length of pre-trained layers: %.f' % (len(pre_dict_ori)))

        model_dict = model.state_dict()
        print('length of new layers: %.f' % (len(model_dict)))
        print('before loading weights: %.12f' % (model.transformer.state_dict()['block1.0.mlp.fc1.weight'].mean()))
        print('before patch_embeddings layer1 weights: %.12f' % (model.transformer.state_dict()['patch_embed2D1.proj.conv.weight'].mean()))
        print('before patch_embeddings layer2 weights: %.12f' % (model.transformer.state_dict()['patch_embed2D2.proj.conv.weight'].mean()))
        print('before position_embeddings weights: %.12f' % (model.transformer.pos_embed2D1.data.mean()))

        for k, v in pre_dict_ori.items():
            if ('pos_embed2D' in k) or ('DecPosEmbed2D' in k): 
                posemb = pre_dict_ori[k]
                posemb_new = model_dict[k]                        

                if posemb.size() == posemb_new.size():
                    print(k+'layer is matched')
                    pre_dict_ori[k] = posemb
                else:
                    ntok_new = posemb_new.size(1)
                    posemb_zoom = ndimage.zoom(posemb[0], (ntok_new / posemb.size(1), 1), order=1)
                    posemb_zoom = np.expand_dims(posemb_zoom, 0)
                    pre_dict_ori[k] = torch.from_numpy(posemb_zoom)         

        pre_dict = {k: v for k, v in pre_dict_ori.items() if k in model_dict}
        print('length of matched layers: %.f' % (len(pre_dict)))
        model_dict.update(pre_dict)
        model.load_state_dict(model_dict)

        print('after loading weights: %.12f' % (model.transformer.state_dict()['block1.0.mlp.fc1.weight'].mean()))
        print('after patch_embeddings layer1 weights: %.12f' % (model.transformer.state_dict()['patch_embed2D1.proj.conv.weight'].mean()))
        print('after patch_embeddings layer2 weights: %.12f' % (model.transformer.state_dict()['patch_embed2D2.proj.conv.weight'].mean()))
        print('after position_embeddings weights: %.12f' % (model.transformer.pos_embed2D1.data.mean()))

    else:
        print('before loading weights: %.12f' % (model.transformer.state_dict()['block1.0.mlp.fc1.weight'].mean()))
        print('before patch_embeddings layer1 weights: %.12f' % (model.transformer.state_dict()['patch_embed2D1.proj.conv.weight'].mean()))
        print('before patch_embeddings layer2 weights: %.12f' % (model.transformer.state_dict()['patch_embed2D2.proj.conv.weight'].mean()))
        print('before position_embeddings weights: %.12f' % (model.transformer.pos_embed2D1.data.mean()))

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=3e-5, momentum=0.99, nesterov=True)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=3e-5, eps=1e-4)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=3e-5, eps=1e-4)

    print(optimizer)

    model.cuda()
    model.train()
    model.float()


    cudnn.benchmark = True

    ############# Load training and validation data
    data_root = 'dataset/'
    data_train_list = args.train_list
    trainloader = data.DataLoader(MyDataSet_seg(data_root, data_train_list, crop_size=(w, h), max_iters=TRAIN_NUM),
                                  batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORK, pin_memory=True)

    data_test_list = args.test_list
    testloader = data.DataLoader(MyValDataSet_seg(data_root, data_test_list, crop_size=(w, h)), batch_size=1, shuffle=False, num_workers=NUM_WORK,
                                pin_memory=True)

    path = NAME
    if not os.path.isdir(path):
        os.makedirs(path)
    f_path = path + 'outputxx.txt'
    f = open(f_path, "a")

    ############# Start the training
    for epoch in range(EPOCH):

        train_loss_dice = []
        train_loss_ce = []
        train_loss_total = []
        train_jac = []

        for i_iter, batch in tqdm(enumerate(trainloader)):

            step = (TRAIN_NUM/BATCH_SIZE)*epoch+i_iter

            images, labels, name = batch
            images = images.cuda()
            labels = torch.tensor(labels).cuda().squeeze(1)

            optimizer.zero_grad()
            lr = adjust_learning_rate(optimizer, step)

            model.train()
            preds = model(images)

            loss_dice, loss_ce = Fuse_Dice_CE(preds, labels)
            term = loss_dice + 1. * loss_ce

            term.backward()

            optimizer.step()

            train_loss_dice.append(loss_dice.cpu().data.numpy())
            train_loss_ce.append(loss_ce.cpu().data.numpy())
            train_loss_total.append(term.cpu().data.numpy())
            train_jac.append(Jaccard(preds, labels))

        line_train = "train_epoch%d: lossTotal=%f, lossDice=%f, lossCE=%f, Jaccard=%f \n" % (epoch, np.nanmean(train_loss_total), np.nanmean(train_loss_dice), np.nanmean(train_loss_ce), np.nanmean(train_jac))
        print(line_train)
        f.write(line_train)

    ############# Save final network
    torch.save(model.state_dict(), path + 'ckp_final.pth')

    ############# Start final testing
    [tdice, tjac, tdice_c, tjac_c, tdice_h, tjac_h, tdice_l, tjac_l] = val_mode_seg(testloader, model, path, epoch)
    line_test = "test_final%d: tdice_ave=%f, tjac_ave=%f, tdice_all=%f, tjac_all=%f, tdice_c=%f, tjac_c=%f, tdice_h=%f, tjac_h=%f, tdice_l=%f, tjac_l=%f \n" % \
            (epoch, (np.nanmean(tdice_c)+np.nanmean(tdice_h)+np.nanmean(tdice_l))/3.0, (np.nanmean(tjac_c)+np.nanmean(tjac_h)+np.nanmean(tjac_l))/3.0, \
            np.nanmean(tdice), np.nanmean(tjac), np.nanmean(tdice_c), np.nanmean(tjac_c), np.nanmean(tdice_h), np.nanmean(tjac_h), np.nanmean(tdice_l), np.nanmean(tjac_l))

    print(line_test)
    f.write(line_test)

if __name__ == '__main__':
    main()
