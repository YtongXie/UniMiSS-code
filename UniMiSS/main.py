import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import utils
import models.MiT as MiTs
from models.MiT import Head

from data_loader3D import Dataset3D
from data_loader2D import Dataset2D


def get_args_parser():
    parser = argparse.ArgumentParser('UniMiSS', add_help=False)

    # Model parameters
    parser.add_argument('--interval', default=2, type=int)
    parser.add_argument('--arch', default='model_small', type=str, choices=['model_tiny', 'model_small'],
                        help="""Name of architecture to train.""")

    parser.add_argument('--img_size2D', default=224, type=int,
                        help="""Size in pixels of input square 2D patches.""")
    parser.add_argument('--img_size3D', default=[16,96,96], type=int,
                        help="""Size in pixels of input square 3D patches.""")

    parser.add_argument('--out_dim', default=65536, type=int,
                        help="""Dimensionality of the UniMiSS head output.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
                        help="""Whether or not to weight normalize the last layer of the UniMiSS head.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float,
                        help="""Base EMA parameter for teacher update.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
                        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
                        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
                        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float,
                        help="""Final value (after linear warmup) of the teacher temperature.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
                        help='Number of warmup epochs for the teacher temperature.')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=False,
                        help="""Whether or not to use half precision for training.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the weight decay.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the weight decay.""")
    parser.add_argument('--clip_grad', type=float, default=0.3,
                        help="""Maximal parameter gradient norm if using gradient clipping.""")
    parser.add_argument('--batch_size_per_gpu', default=4, type=int,
                        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=200, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int,
                        help="""Number of epochs during which we keep the output layer fixed. 
                        Typically doing so during the first epoch helps training. 
                        Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0008, type=float,
                        help="""Learning rate at the end of linear warmup (highest LR used during training).""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
                        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help="""Target LR at the end of optimization.""")
    parser.add_argument('--optimizer', default='adamw', type=str, choices=['adamw', 'sgd', 'lars'],
                        help="""Type of optimizer.""")

    # Others
    parser.add_argument('--data_path', default='../data/', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--list_path2D', default='2D_images.txt', type=str)
    parser.add_argument('--list_path3D', default='3D_images.txt', type=str)
    parser.add_argument('--output_dir', default="snapshots", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=12, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    return parser


def train_func(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    train_set2D = Dataset2D(args.data_path, args.list_path2D, crop_size_2D=args.img_size2D)
    train_set3D = Dataset3D(args.data_path, args.list_path3D, crop_size_3D=args.img_size3D)

    data_loader2D = torch.utils.data.DataLoader(
        train_set2D,
        sampler=torch.utils.data.DistributedSampler(train_set2D, shuffle=True),
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    data_loader3D = torch.utils.data.DataLoader(
        train_set3D,
        sampler=torch.utils.data.DistributedSampler(train_set3D, shuffle=True),
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    print(f"Data loaded: there are {len(train_set3D)+len(train_set2D)} images.")

    # ============ building student and teacher networks ... ============
    student = MiTs.__dict__[args.arch](
        norm2D='IN2',
        norm3D='IN3',
        act='LeakyReLU',
        ws=False,
        img_size2D=args.img_size2D,
        img_size3D=args.img_size3D,
        modal_type='MM',
        drop_path_rate=0.1,  # stochastic depth
    )
    teacher = MiTs.__dict__[args.arch](
        norm2D='IN2',
        norm3D='IN3',
        act='LeakyReLU',
        ws=False,
        img_size2D=args.img_size2D,
        img_size3D=args.img_size3D,
        modal_type='MM')

    embed_dim = student.embed_dim

    student = utils.ModuleWrapper(student,Head(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ))
    teacher = utils.ModuleWrapper(
        teacher,
        Head(embed_dim, args.out_dim, args.use_bn_in_head),
    )
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher

    # DDP wrapper...
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    trainloss = TrainLoss(
        args.out_dim,
        2,
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with UniMiSS
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader3D.dataset),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader3D.dataset),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1, args.epochs, len(data_loader3D.dataset))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        loss=trainloss,
    )

    start_epoch = to_restore["epoch"]

    interval = args.interval
    for epoch in range(start_epoch, args.epochs, interval*2):

        train_several_epoch(student, teacher, teacher_without_ddp, trainloss, data_loader2D, epoch, epoch+interval,
            optimizer, lr_schedule, wd_schedule, momentum_schedule, fp16_scaler, '2D', args)

        train_several_epoch(student, teacher, teacher_without_ddp, trainloss, data_loader3D, epoch+interval, epoch+interval*2,
                        optimizer, lr_schedule, wd_schedule, momentum_schedule, fp16_scaler, '3D', args)



def train_several_epoch(student, teacher, teacher_without_ddp, trainloss, data_loader, start_epoch, end_epoch,
                        optimizer, lr_schedule, wd_schedule, momentum_schedule, fp16_scaler, modal_type, args):

    start_time = time.time()
    print("Starting " + modal_type + " UniMiSS training !")
    for epoch in range(start_epoch, end_epoch):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of UniMiSS ... ============
        if modal_type == '2D': 

            for name, param in student.module.named_parameters():  
                if '3D' in name:  
                    param.requires_grad = False
            student = nn.parallel.DistributedDataParallel(student.module, device_ids=[args.gpu])

            train_stats = train_one_epoch2D(student, teacher, teacher_without_ddp, trainloss,
                data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
                epoch, fp16_scaler, args)
        else:

            for name, param in student.module.named_parameters():  
                if '3D' in name:  
                    param.requires_grad = True
            student = nn.parallel.DistributedDataParallel(student.module, device_ids=[args.gpu])

            train_stats = train_one_epoch3D(student, teacher, teacher_without_ddp, trainloss,
                data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
                epoch, fp16_scaler, args)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'loss': trainloss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f: f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training ' + modal_type + ' time {}'.format(total_time_str))



def train_one_epoch2D(student, teacher, teacher_without_ddp, trainloss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch, fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, subjects_batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader.dataset) * epoch + it
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move 2D images to gpu
        images = [im.cuda(non_blocking=True) for im in subjects_batch]

        # teacher and student forward passes + compute 2D loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(images)
            student_output = student(images)
            
            loss = trainloss(student_output, teacher_output, epoch)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss2D=loss.item())
        metric_logger.update(lr2D=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd2D=optimizer.param_groups[0]["weight_decay"])


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



def train_one_epoch3D(student, teacher, teacher_without_ddp, trainloss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch, fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, subjects_batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader.dataset) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move 3D images to gpu
        images = [im.cuda(non_blocking=True) for im in subjects_batch]

        #3D volume to 2D slices
        image_slices = []
        image_slices.append(images[0].view(-1, images[0].size()[-2], images[0].size()[-1]).unsqueeze(1).repeat_interleave(3, 1))
        image_slices.append(images[1].view(-1, images[1].size()[-2], images[1].size()[-1]).unsqueeze(1).repeat_interleave(3, 1))

        # teacher and student forward passes + compute 3D loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(images[:2]) # volume
            teacher_output_slices = teacher(image_slices) # slices
            teacher_output_slices = teacher_output_slices.view(images[0].size()[0] * 2, images[0].size()[2], -1).mean(1)

            student_output = student(images) # volume
            student_output_slices = student(image_slices) # slices
            student_output_slices = student_output_slices.view(images[0].size()[0] * 2, images[0].size()[2], -1).mean(1)

            volume_loss = trainloss(student_output, teacher_output, epoch)  # volume loss only
            slices_volume_loss = trainloss(student_output_slices, teacher_output, epoch)  # slices --> volume loss

            slices_loss = trainloss(student_output_slices, teacher_output_slices, epoch)  # slices loss only
            volume_slices_loss = trainloss(student_output, teacher_output_slices, epoch)  # volume --> slices loss

            loss = volume_loss + slices_volume_loss + slices_loss + volume_slices_loss

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss_volume=volume_loss.item())
        metric_logger.update(loss_slice=slices_loss.item())
        metric_logger.update(loss_SV=slices_volume_loss.item())
        metric_logger.update(loss_VS=volume_slices_loss.item())
        metric_logger.update(lr3D=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd3D=optimizer.param_groups[0]["weight_decay"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



class TrainLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('UniMiSS', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_func(args)
