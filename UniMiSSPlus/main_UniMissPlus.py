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
import MiTplus as nets
from MiTplus import DINOHead
from data_loader3D2D import Dataset3D2D
from data_loader2D import Dataset2D


def get_args_parser():
    parser = argparse.ArgumentParser('UniMissPlus', add_help=False)
    parser.add_argument('--interval3d', default=1, type=int)
    parser.add_argument('--interval2d', default=1, type=int)

    # Model parameters
    parser.add_argument('--arch', default='MiTplus', type=str, help="""Name of architecture to train.""")
    parser.add_argument('--weights_s', type=float, default=1.0, help="""loss weight for slice2CT.""")
    parser.add_argument('--weights_drr', type=float, default=1.0, help="""loss weight for DRR2CT.""")
    parser.add_argument('--weights_re', type=float, default=1.0, help="""loss weight for re.""")

    parser.add_argument('--img_size2D', default=96, type=int, help="""Size in pixels of input 2D patches.""")
    parser.add_argument('--img_size3D', default=[16,96,96], type=int, help="""Size in pixels of input 3D patches.""")

    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with deit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=False, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=8, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=2, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.5, 0.7),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=0, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.8, 1.0),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--data_path', default='data/', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--list_path2D', default='2D_images.txt', type=str)
    parser.add_argument('--list_path3D', default='3D_images.txt', type=str)
    parser.add_argument('--output_dir', default="snapshots/tmp", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=100, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    return parser


def train_dino(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============

    train_set2D = Dataset2D(args.data_path, args.list_path2D, max_iters=args.epochs, crop_size_2D=args.img_size2D, local_crops_number=args.local_crops_number)
    train_set3D = Dataset3D2D(args.data_path, args.list_path3D, crop_size_2D=args.img_size2D, crop_size_3D=args.img_size3D)

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

    data_loader2D = iter(data_loader2D)

    print(f"Data loaded: there are {len(train_set3D)} images.")

    # ============ building student and teacher networks ... ============
    # if the network is a vision transformer (i.e. deit_tiny, deit_small, vit_base)
    if args.arch in nets.__dict__.keys():
        student = nets.__dict__[args.arch](
            img_size2D=args.img_size2D,
            img_size3D=args.img_size3D,
            modal_type='MM',
            drop_path_rate=0.1
        )
        teacher = nets.__dict__[args.arch](
            img_size2D=args.img_size2D,
            img_size3D=args.img_size3D,
            modal_type='MM')

        embed_dim = student.transformer.embed_dims[-1]
        
    print(student)

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper_re(student, DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ))
    teacher = utils.MultiCropWrapper_re(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
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
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    mse_loss = nn.MSELoss()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
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
        args.epochs, len(data_loader3D),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader3D),
    )
    print(len(data_loader2D))
    print(len(data_loader3D))
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1, args.epochs, len(data_loader3D))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "UniMissPlus.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )

    args.data_loader3D = data_loader3D

    start_epoch = to_restore["epoch"]

    for epoch in range(start_epoch, args.epochs, args.interval3d+args.interval2d):
        # print(epoch)

        train_several_epoch(student, teacher, teacher_without_ddp, dino_loss, mse_loss, data_loader3D, epoch, epoch+args.interval3d,  
                        optimizer, lr_schedule, wd_schedule, momentum_schedule, fp16_scaler, '3D', args)
                    
        train_several_epoch(student, teacher, teacher_without_ddp, dino_loss, mse_loss, data_loader2D, epoch+args.interval3d, epoch+args.interval3d+args.interval2d,
            optimizer, lr_schedule, wd_schedule, momentum_schedule, fp16_scaler, '2D', args)


def train_several_epoch(student, teacher, teacher_without_ddp, dino_loss, mse_loss, data_loader, start_epoch, end_epoch, 
                        optimizer, lr_schedule, wd_schedule, momentum_schedule, fp16_scaler, modal_type, args):

    start_time = time.time()
    print("Starting " + modal_type + "UniMissPlus training !")
    for epoch in range(start_epoch, end_epoch):
        
        # ============ training one epoch of DINO ... ============
        if modal_type == '2D': 
            # data_loader.sampler.set_epoch(epoch)

            for name, param in student.module.named_parameters():  
                if '3D' in name:  
                    param.requires_grad = False
            student = nn.parallel.DistributedDataParallel(student.module, device_ids=[args.gpu])

            train_stats = train_one_epoch2D(student, teacher, teacher_without_ddp, dino_loss, mse_loss,
                data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
                epoch, fp16_scaler, args)
        else:
            data_loader.sampler.set_epoch(epoch)

            for name, param in student.module.named_parameters():  
                if '3D' in name:  
                    param.requires_grad = True
            student = nn.parallel.DistributedDataParallel(student.module, device_ids=[args.gpu])

            train_stats = train_one_epoch3D(student, teacher, teacher_without_ddp, dino_loss, mse_loss,
                data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
                epoch, fp16_scaler, args)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'UniMissPlus.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f: f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training ' + modal_type + ' time {}'.format(total_time_str))



def train_one_epoch2D(student, teacher, teacher_without_ddp, dino_loss, mse_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch, fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, _ in enumerate(metric_logger.log_every_add(args.data_loader3D, 10, header)):
        # update weight decay and learning rate according to their schedule
        # it += 1
        it = len(args.data_loader3D) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in data_loader.next()]

        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            # teacher_encoder = teacher(images) 
            # student_encoder = student(images)
            _, teacher_encoder = teacher(images) 
            student_decoder, student_encoder = student(images)

            # loss = dino_loss(student_encoder, teacher_encoder, epoch)
            loss_un = dino_loss(student_encoder, teacher_encoder, epoch)
            loss_re = mse_loss(torch.sigmoid(student_decoder), torch.cat(images))
            loss = loss_un + args.weights_re * loss_re

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
        metric_logger.update(loss_un2D=loss_un.item())
        metric_logger.update(loss_re2D=loss_re.item())
        metric_logger.update(lr2D=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd2D=optimizer.param_groups[0]["weight_decay"])

        # if it % 50 == 0:
        #     torch.cuda.empty_cache()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



def train_one_epoch3D(student, teacher, teacher_without_ddp, dino_loss, mse_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch, fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, subjects_batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        # it += 1
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images2D = [im.cuda(non_blocking=True) for im in subjects_batch[0]]
        images3D = [im.cuda(non_blocking=True) for im in subjects_batch[1]]

        image3D_slices = []
        image3D_slices.append(images3D[0][:,:,::2].view(-1, images3D[0].size()[-2], images3D[0].size()[-1]).unsqueeze(1).repeat_interleave(3, 1))
        image3D_slices.append(images3D[1][:,:,::2].view(-1, images3D[1].size()[-2], images3D[1].size()[-1]).unsqueeze(1).repeat_interleave(3, 1))

        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):

            _, teacher_output_CT = teacher(images3D) 

            student_decoder_CT, student_output_CT = student(images3D)
            student_decoder_XR, student_output_XR = student(images2D)
            _, student_output3D_slices = student(image3D_slices)
            student_output3D_slices = student_output3D_slices.view(images3D[0].size()[0] * 2, images3D[0].size()[2]//2, -1).mean(1)

            loss_CT2CT = dino_loss(student_output_CT, teacher_output_CT, epoch)
            loss_XR2CT = dino_loss(student_output_XR, teacher_output_CT, epoch)
            loss_slice2CT = dino_loss(student_output3D_slices, teacher_output_CT, epoch)

            inputs_nor_CT = torch.cat(images3D).clone()
            inputs_nor_CT -= inputs_nor_CT.min(-1,keepdim=True)[0].min(-2,keepdim=True)[0].min(-3,keepdim=True)[0]
            inputs_nor_CT /= inputs_nor_CT.max(-1,keepdim=True)[0].max(-2,keepdim=True)[0].max(-3,keepdim=True)[0]
            loss_re_CT = mse_loss(torch.sigmoid(student_decoder_CT), inputs_nor_CT)
            loss_re_XR = mse_loss(torch.sigmoid(student_decoder_XR), torch.cat(images2D))

            loss = loss_CT2CT + args.weights_drr * loss_XR2CT + args.weights_s * loss_slice2CT + args.weights_re * loss_re_CT + args.weights_re * loss_re_XR

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
        metric_logger.update(loss_3Dall=loss.item())
        metric_logger.update(loss_CT2CT=loss_CT2CT.item())
        metric_logger.update(loss_XR2CT=loss_XR2CT.item())
        metric_logger.update(loss_slice2CT=loss_slice2CT.item())
        metric_logger.update(loss_re_CT=loss_re_CT.item())
        metric_logger.update(loss_re_XR=loss_re_XR.item())
        metric_logger.update(lr3D=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd3D=optimizer.param_groups[0]["weight_decay"])

        # if it % 50 == 0:
        #     torch.cuda.empty_cache()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



class DINOLoss(nn.Module):
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
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_dino(args)