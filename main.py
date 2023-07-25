"""
Training script of SgMg
Modified from DETR (https://github.com/facebookresearch/detr)
"""
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.cuda.amp as amp

import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import train_one_epoch, evaluate, evaluate_a2d
from models import build_model

from utils import pre_trained_model_to_finetune
from util.logger import TensorboardLogger
import opts

import warnings
warnings.filterwarnings("ignore")

def main(args):
    args.masks = True
    args.binary = True  # only run on binary referred for joint

    utils.init_distributed_mode(args)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "configs"), 'w') as f:
        f.write(str(args) + '\n')
    print("Record configs finish.")
    print(f'\n **** Run on {args.dataset_file} dataset. **** \n')

    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Logger
    local_rank = torch.distributed.get_rank()
    logger = None
    if local_rank == 0:
        long_id = args.exp_name
        logger = TensorboardLogger(long_id, long_id, local_rank)  # id name + time tag
        logger.log_string('hyperpara', str(args))

    model, criterion, postprocessor = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_text_encoder_names) 
                 and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_text_encoder_names) and p.requires_grad],
            "lr": args.lr_text_encoder,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_drop)
    grad_scaler = amp.GradScaler(enabled=args.amp)
    print("\n **** Using AMP? {}. **** \n".format(args.amp))

    # Do not load when evaluating a2d or jhmdb.
    if not (args.eval and (args.dataset_file == 'a2d' or args.dataset_file == 'jhmdb')):
        dataset_train = build_dataset(args.dataset_file, image_set='train', args=args)
        if args.distributed:
            if args.cache_mode:
                sampler_train = samplers.NodeDistributedSampler(dataset_train)
            else:
                sampler_train = samplers.DistributedSampler(dataset_train)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, args.batch_size, drop_last=True)
        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                       collate_fn=utils.collate_fn, num_workers=args.num_workers)

    # A2D-Sentences
    if args.dataset_file == 'a2d' or args.dataset_file == 'jhmdb':
        dataset_val = build_dataset(args.dataset_file, image_set='val', args=args)
        if args.distributed:
            if args.cache_mode:
                sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
            else:
                sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                     drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers, pin_memory=True)

    if args.dataset_file == "jhmdb":
        assert args.resume is not None, "Please provide the checkpoint to resume for JHMDB-Sentences"
        print("============================================>")
        print("JHMDB-Sentences are directly evaluated using the checkpoint trained on A2D-Sentences")
        print("Load checkpoint weights from {} ...".format(args.pretrained_weights))
        print("============================================>")

    # *** for refer_youtube_vos and A2D-Sentences, finetune using the pretrained weights on Ref-COCO ***
    if args.dataset_file != "davis" and args.dataset_file != "jhmdb" and args.pretrained_weights is not None:
        print("============================================>")
        print("Load pretrained weights from {} ...".format(args.pretrained_weights))
        checkpoint = torch.load(args.pretrained_weights, map_location="cpu")
        checkpoint_dict = pre_trained_model_to_finetune(checkpoint, args)
        model_without_ddp.load_state_dict(checkpoint_dict, strict=False)
        print("============================================>")

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print("\n **** Missing Keys: {}. **** \n".format(missing_keys))
        if len(unexpected_keys) > 0:
            print("\n **** Unexpected Keys: {}. **** \n".format(unexpected_keys))
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            checkpoint['lr_scheduler'].pop('gamma')
            checkpoint['lr_scheduler'].pop('milestones')
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # this is a hack for doing experiment that 【resume from checkpoint【 and 【also modify lr scheduler (e.g., decrease lr in advance).
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            if 'grad_scaler' in checkpoint:
                grad_scaler.load_state_dict(checkpoint['grad_scaler'])
            args.start_epoch = checkpoint['epoch'] + 1
        print("\n **** Loaded previous checkpoint from {}. **** \n".format(args.resume))

    # evaluation of a2d or jhmdb
    if args.eval:
        assert args.dataset_file == 'a2d' or args.dataset_file == 'jhmdb', \
                    'Only A2D-Sentences and JHMDB-Sentences datasets support evaluation'
        print("\n **** Begin to evaluating {}. **** \n".format(args.dataset_file))
        with torch.no_grad():
            test_stats = evaluate_a2d(model, data_loader_val, postprocessor, device, args)
        return

    print("\n **** Start training, total poch is: {}, begin from epoch: {}. **** \n".format(args.epochs, args.start_epoch))
    start_time = time.time()
    total_itr_num = 0
    for epoch in range(args.start_epoch, args.epochs):
        if epoch > 0 and not (args.eval and (args.dataset_file == 'a2d' or args.dataset_file == 'jhmdb')):
            # ****************** Reload dataset ******************
            args.current_epoch = epoch
            dataset_train = build_dataset(args.dataset_file, image_set='train', args=args)
            if args.distributed:
                if args.cache_mode:
                    sampler_train = samplers.NodeDistributedSampler(dataset_train)
                else:
                    sampler_train = samplers.DistributedSampler(dataset_train)
            else:
                sampler_train = torch.utils.data.RandomSampler(dataset_train)

            batch_sampler_train = torch.utils.data.BatchSampler(
                sampler_train, args.batch_size, drop_last=True)
            data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                           collate_fn=utils.collate_fn, num_workers=args.num_workers)
            print("Reload dataset.")
            # ******************************************************

        if args.distributed:
            sampler_train.set_epoch(epoch)

        epoch_s_ = time.time()
        train_stats, total_itr_num = train_one_epoch(
            args, model, criterion, data_loader_train, optimizer, grad_scaler, device, epoch,
            args.clip_max_norm, total_itr_num, lr_scheduler, logger)
        epoch_e_ = time.time()
        print("\n **** Train one epoch time cost is {}h. **** \n".format((epoch_e_-epoch_s_)/3600))

        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 1 == 0:
            checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'grad_scaler': grad_scaler.state_dict(),
                }, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        # if args.dataset_file == 'a2d':
        #     print("Begin to evaluating {}...".format(args.dataset_file))
        #     with torch.no_grad():
        #         test_stats = evaluate_a2d(model, data_loader_val, postprocessor, device, args)
        #     log_stats.update({**{f'{k}': v for k, v in test_stats.items()}})

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('\n **** Total training time for this task is {}. **** \n'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SgMg training and evaluation script', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    gpu_num = torch.cuda.device_count()
    print("Use GPU number is: ", gpu_num)
    if args.pretrained_weights is None:
        args.lr *= gpu_num / 4
        args.lr_backbone *= gpu_num / 4
        args.lr_text_encoder *= gpu_num / 4
    else:
        args.lr *= gpu_num / 8
        args.lr_backbone *= gpu_num / 8
        args.lr_text_encoder *= gpu_num / 8
    print("\n **** After adjust with GPU&BATCH num {}/{}, lr: {}, lr_backbone: {}, lr_text_backbone: {}. **** \n".format(gpu_num, args.batch_size, args.lr, args.lr_backbone, args.lr_text_encoder))
    main(args)



