import torch
import numpy as np
import torch.backends.cudnn as cudnn
import os
from tqdm import tqdm
from net.MiTPlus_encoder import MiTPlus_encoder
import matplotlib.pyplot as plt
from dataset.my_datasets import MyDataSet_cls, MyTestDataSet_cls
from torch.utils import data
import argparse
from sklearn import metrics
from scipy import ndimage

parser = argparse.ArgumentParser()

parser.add_argument("-train_list", type=str, default='dataset/CXR_Covid-19/CXR_Covid-19_Challenge_train_all.txt')
parser.add_argument("-test_list", type=str, default='dataset/CXR_Covid-19/CXR_Covid-19_Challenge_test.txt')

parser.add_argument("-GPU", type=str, default='0')
parser.add_argument("-INPUT_SIZE", type=str, default='224, 224')
parser.add_argument("-NUM_CLASSES", type=int, default=3, help='number of epoches')
parser.add_argument("-BATCH_SIZE", type=int, default=32, help='number of epoches')
parser.add_argument("-EPOCH", type=int, default=100, help='number of epoches')
parser.add_argument("-LEARNING_RATE", type=float, default=0.001, help='number of learning rate')
parser.add_argument("-optimizer", type=str, default='AdamW', help='type of optimizer')

parser.add_argument("-deterministic", default=True, help="use this if you want to use deterministic")

parser.add_argument("-save_path", type=str, default='xxx/', help='save path')
parser.add_argument("-pre_train", default=False, help="use this if you want to load pretrained weights")
parser.add_argument("-pre_train_path", type=str, default='UniMiss.pth', help='pretrained path')

parser.add_argument("-num_works", type=int, default=8, help="num of workers")
parser.add_argument("-seed", type=int, default=1234, help='number of seed')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=args.GPU
INPUT_SIZE = args.INPUT_SIZE
w, h = map(int, INPUT_SIZE.split(','))
NUM_CLASSES = args.NUM_CLASSES
BATCH_SIZE = args.BATCH_SIZE
EPOCH = args.EPOCH

deterministic = args.deterministic

NAME = args.save_path
pre_train = args.pre_train
pre_train_path = args.pre_train_path

LEARNING_RATE = args.LEARNING_RATE
MOMENTUM = 0.9
POWER = 0.9
WEIGHT_DECAY = 3e-5

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


def adjust_learning_rate(optimizer, all_STEPS, i_iter):
    lr = lr_poly(LEARNING_RATE, i_iter, all_STEPS, POWER)
    optimizer.param_groups[0]['lr'] = lr
    return lr

def cla_evaluate(label, binary_score, pro_score):
    acc = metrics.accuracy_score(label, binary_score)
    AP = metrics.average_precision_score(label, pro_score)
    auc = metrics.roc_auc_score(label, pro_score)
    return acc, auc, AP

def test_mode_cls(dataloader, model):
    # test
    pro_score = []
    label_val = []
    for index, batch in tqdm(enumerate(dataloader)):
        data, label = batch
        bs, n_crops, c, h, w =  data.size() 
        data = data.view(-1, c, h, w).cuda()

        model.eval()
        with torch.no_grad():

            pred = model(data)
            pred1 = model(torch.flip(data, [-1]))
            pred2 = model(torch.flip(data, [-2]))
            pred3 = model(torch.flip(data, [-1, -2]))

            pred = (torch.softmax(pred, dim=1) + torch.softmax(pred1, dim=1) + torch.softmax(pred2, dim=1) + torch.softmax(pred3, dim=1)) / 4.

        pred = pred.view(bs, n_crops, -1).mean(1)
        pro_score.append(pred.cpu().data.numpy())
        label_val.append(label.data.numpy())

    pro_score = np.concatenate(pro_score) 
    binary_score = np.eye(3)[np.argmax(np.array(pro_score), axis=-1)]

    label_val = np.concatenate(label_val)
    label_val = np.eye(3)[np.int64(np.array(label_val))]

    label_val_a = label_val
    pro_score_a = pro_score
    binary_score_a = binary_score
    val_acc, val_auc, val_AP = cla_evaluate(label_val_a, binary_score_a, pro_score_a)

    return val_acc, val_auc, val_AP



def main():
    """Create the network and start the training."""

    cudnn.enabled = True

    model = MiTPlus_encoder(num_classes=NUM_CLASSES)
    # for name, param in model.named_parameters():
    #     if 'head' not in name:
    #         param.requires_grad = False

    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of Network Params: %.2f(e6)' % (total / 1e6))
    # print(model)

    if pre_train:
        pre_type = 'student'  #teacher student
        print('*********loading from checkpoint ssl: {}'.format(pre_train_path))
        print('*********loading from checkpoint ssl --> epoch: {}'.format(str(torch.load(pre_train_path, map_location="cpu")['epoch'])))

        if pre_type == 'teacher': 
            pre_dict_ori = torch.load(pre_train_path, map_location="cpu")[pre_type]
            pre_dict_ori = {k.replace("backbone.", ""): v for k, v in pre_dict_ori.items()}
            print('Teacher: length of pre-trained layers: %.f' % (len(pre_dict_ori)))
        elif pre_type == 'student': 
            pre_dict_ori = torch.load(pre_train_path, map_location="cpu")[pre_type]
            pre_dict_ori = {k.replace("module.backbone.", ""): v for k, v in pre_dict_ori.items()}
            print('Student: length of pre-trained layers: %.f' % (len(pre_dict_ori)))

        pre_dict_ori = {k.replace("transformer.", ""): v for k, v in pre_dict_ori.items()}

        model_dict = model.state_dict()
        print('length of new layers: %.f' % (len(model_dict)))
        print('before loading weights: %.12f' % (model.state_dict()['block1.0.mlp.fc1.weight'].mean()))
        print('before patch_embeddings layer1 weights: %.12f' % (model.state_dict()['patch_embed2D1.proj.conv.weight'].mean()))
        print('before patch_embeddings layer2 weights: %.12f' % (model.state_dict()['patch_embed2D2.proj.conv.weight'].mean()))
        print('before position_embeddings weights: %.12f' % (model.pos_embed2D1.data.mean()))

        for k, v in pre_dict_ori.items():
            if 'pos_embed2D' in k: 
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

        print('after loading weights: %.12f' % (model.state_dict()['block1.0.mlp.fc1.weight'].mean()))
        print('after patch_embeddings layer1 weights: %.12f' % (model.state_dict()['patch_embed2D1.proj.conv.weight'].mean()))
        print('after patch_embeddings layer2 weights: %.12f' % (model.state_dict()['patch_embed2D2.proj.conv.weight'].mean()))
        print('after position_embeddings weights: %.12f' % (model.pos_embed2D1.data.mean()))

    else:
        print('before loading weights: %.12f' % (model.state_dict()['block1.0.mlp.fc1.weight'].mean()))
        print('before patch_embeddings layer1 weights: %.12f' % (model.state_dict()['patch_embed2D1.proj.conv.weight'].mean()))
        print('before patch_embeddings layer2 weights: %.12f' % (model.state_dict()['patch_embed2D2.proj.conv.weight'].mean()))
        print('before position_embeddings weights: %.12f' % (model.pos_embed2D1.data.mean()))


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
    data_root = 'dataset/CXR_Covid-19/'
    data_train_list = args.train_list
    trainloader = data.DataLoader(MyDataSet_cls(data_root, data_train_list), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORK, pin_memory=True)

    data_test_list = args.test_list
    testloader = data.DataLoader(MyTestDataSet_cls(data_root, data_test_list), batch_size=3, shuffle=False, num_workers=NUM_WORK, pin_memory=True)


    path = NAME
    if not os.path.isdir(path):
        os.makedirs(path)
    f_path = path + 'outputxx.txt'

    ############# Start the training
    for epoch in range(EPOCH):

        train_loss_total = []

        for i_iter, batch in tqdm(enumerate(trainloader)):

            step = len(trainloader)*epoch+i_iter

            images, labels = batch
            images = images.cuda()
            labels = labels.cuda().long()

            optimizer.zero_grad()
            lr = adjust_learning_rate(optimizer, len(trainloader)*EPOCH, step)
            model.train()
            preds = model(images)

            term = loss_cls_CE(preds, labels)

            term.backward()
            optimizer.step()

            train_loss_total.append(term.cpu().data.numpy())

        print("train_epoch%d: lossTotal=%f \n" % (epoch, np.nanmean(train_loss_total)))

        ############# Save final network
        torch.save(model.state_dict(), path + 'ckp_weights.pth')

    f = open(f_path, "a")
    ############# Start final testing
    [test_acc, test_auc, test_AP] = test_mode_cls(testloader, model)
    line_test = "test_final%d: tacc=%f, tauc=%f, tAP=%f \n" % (epoch, test_acc, test_auc, test_AP)
    print(line_test)
    f.write(line_test)


if __name__ == '__main__':
    main()
