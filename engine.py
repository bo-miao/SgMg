"""
Train and eval functions used in main.py
Modified from DETR (https://github.com/facebookresearch/detr)
"""
import math
import time

from models import postprocessors
import os
import sys
from typing import Iterable

import torch
import torch.distributed as dist
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.refexp_eval import RefExpEvaluator
import torch.cuda.amp as amp
import random
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image
import numpy as np
from datasets.a2d_eval import calculate_precision_at_k_and_iou_metrics, calculate_bbox_precision_at_k_and_iou_metrics


def train_one_epoch(args, model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    grad_scaler: torch.cuda.amp.GradScaler,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    total_itr_num=0, lr_scheduler=None, logger=None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    itr_counter = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        itr_counter += 1
        total_itr_num += 1

        samples = samples.to(device)
        captions = [t["caption"] for t in targets]

        targets = utils.targets_to(targets, device)
        with amp.autocast(enabled=args.amp):
            outputs = model(samples, captions, targets)
            loss_dict = criterion(outputs, targets)
            for k in loss_dict:
                if torch.isnan(loss_dict[k]):
                    print("loss {} is Nan!!!!".format(k))
                    for kk in loss_dict:
                        if not torch.isnan(loss_dict[kk]):
                            loss_dict[k] = loss_dict[kk] - loss_dict[kk]
                            break

            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()
        if logger is not None and (total_itr_num + itr_counter) % 200 == 0:
            logger.log_scalar('train/lr', lr_scheduler.get_last_lr()[0], total_itr_num + itr_counter)
            logger.log_scalar('train/total_loss', losses.item(), total_itr_num + itr_counter)
            logger.add_dict(loss_dict, total_itr_num + itr_counter)

        if not math.isfinite(loss_value):
            print("\n **** Loss is {}, stopping training. **** \n".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if args.amp:
            grad_scaler.scale(losses).backward()
            if max_norm > 0:
                grad_scaler.unscale_(optimizer)
                grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, error_if_nonfinite=False)
            else:
                grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            losses.backward()
            if max_norm > 0:
                grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, error_if_nonfinite=False)
            else:
                grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
            optimizer.step()

        # metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, total_itr_num


@torch.no_grad()
def evaluate_a2d(model, data_loader, postprocessor, device, args):
    model.eval()
    predictions = []
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    for samples, targets in metric_logger.log_every(data_loader, 100, header):
        image_ids = [t['image_id'] for t in targets]

        samples = samples.to(device)
        captions = [t["caption"] for t in targets]
        targets = utils.targets_to(targets, device)

        outputs = model(samples, captions, targets)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        processed_outputs = postprocessor(outputs, orig_target_sizes, target_sizes)

        for p, image_id in zip(processed_outputs, image_ids):
            for s, m in zip(p['scores'], p['rle_masks']):
                    predictions.append({'image_id': image_id,
                                        'category_id': 1,  # dummy label, as categories are not predicted in ref-vos
                                        'segmentation': m,
                                        'score': s.item()})
    
    # gather and merge predictions from all gpus
    gathered_pred_lists = utils.all_gather(predictions)
    predictions = [p for p_list in gathered_pred_lists for p in p_list]
    # evaluation
    eval_metrics = {}
    if utils.is_main_process():
        if args.dataset_file == 'a2d':
            coco_gt = COCO(os.path.join(args.a2d_path, 'a2d_sentences_test_annotations_in_coco_format.json'))
        elif args.dataset_file == 'jhmdb':
            coco_gt = COCO(os.path.join(args.jhmdb_path, 'jhmdb_sentences_gt_annotations_in_coco_format.json'))
        else:
            raise NotImplementedError
        coco_pred = coco_gt.loadRes(predictions)
        coco_eval = COCOeval(coco_gt, coco_pred, iouType='segm')
        coco_eval.params.useCats = 0  # ignore categories as they are not predicted in ref-vos task
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        ap_labels = ['mAP 0.5:0.95', 'AP 0.5', 'AP 0.75', 'AP 0.5:0.95 S', 'AP 0.5:0.95 M', 'AP 0.5:0.95 L']
        ap_metrics = coco_eval.stats[:6]
        eval_metrics = {l: m for l, m in zip(ap_labels, ap_metrics)}
        # Precision and IOU
        precision_at_k, overall_iou, mean_iou = calculate_precision_at_k_and_iou_metrics(coco_gt, coco_pred)
        eval_metrics.update({f'P@{k}': m for k, m in zip([0.5, 0.6, 0.7, 0.8, 0.9], precision_at_k)})
        eval_metrics.update({'overall_iou': overall_iou, 'mean_iou': mean_iou})
        print(eval_metrics)

    # sync all processes before starting a new epoch or exiting
    dist.barrier()
    return eval_metrics


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, evaluator_list, device, args):
    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    iou_sum = 0.
    iou_count = 0.
    predictions = []
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        dataset_name = targets[0]["dataset_name"]
        samples = samples.to(device)
        captions = [t["caption"] for t in targets]
        image_ids = [t['image_id'] for t in targets]
        targets = utils.targets_to(targets, device)
        # forward
        outputs = model(samples, captions, targets)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)  # origin size before resize
        # process box and mask results
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)  # rshaped size with min len 360
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)  # [[q,1,h,w], ... xb]

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        for evaluator in evaluator_list:
            evaluator.update(res)

        # REC & RES predictions
        for p, target in zip(results, targets):
            for s, b, m in zip(p['scores'], p['boxes'], p['rle_masks']):
                    predictions.append({'image_id': target['image_id'].item(),
                                        'category_id': 1,  # dummy label, as categories are not predicted in ref-vos
                                        'bbox': b.tolist(),
                                        'segmentation': m,
                                        'score': s.item()})

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    for evaluator in evaluator_list:
        evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    refexp_res = None
    for evaluator in evaluator_list:
        if isinstance(evaluator, CocoEvaluator):
            evaluator.accumulate()
            evaluator.summarize()
        elif isinstance(evaluator, RefExpEvaluator):
            refexp_res = evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    # update stats
    for evaluator in evaluator_list:
        if isinstance(evaluator, CocoEvaluator):
            if "bbox" in postprocessors.keys():
                stats["coco_eval_bbox"] = evaluator.coco_eval["bbox"].stats.tolist()
            if "segm" in postprocessors.keys():
                stats["coco_eval_masks"] = evaluator.coco_eval["segm"].stats.tolist()
    if refexp_res is not None:
        stats.update(refexp_res)

    # evaluate RES
    # gather and merge predictions from all gpus
    gathered_pred_lists = utils.all_gather(predictions)
    predictions = [p for p_list in gathered_pred_lists for p in p_list]

    eval_metrics = {}
    if utils.is_main_process():
        if dataset_name == 'refcoco':
            coco_gt = COCO(os.path.join(args.coco_path, 'refcoco/instances_refcoco_val.json'))
        elif dataset_name == 'refcoco+':
            coco_gt = COCO(os.path.join(args.coco_path, 'refcoco+/instances_refcoco+_val.json'))
        elif dataset_name == 'refcocog':
            coco_gt = COCO(os.path.join(args.coco_path, 'refcocog/instances_refcocog_val.json'))
        else:
            raise NotImplementedError
        coco_pred = coco_gt.loadRes(predictions)
        coco_eval = COCOeval(coco_gt, coco_pred, iouType='segm')
        coco_eval.params.useCats = 0  # ignore categories as they are not predicted in ref-vos task
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        # ap_labels = ['mAP 0.5:0.95', 'AP 0.5', 'AP 0.75', 'AP 0.5:0.95 S', 'AP 0.5:0.95 M', 'AP 0.5:0.95 L']
        # ap_metrics = coco_eval.stats[:6]
        # eval_metrics = {l: m for l, m in zip(ap_labels, ap_metrics)}
        # Precision and IOU
        # bbox
        precision_at_k, overall_iou, mean_iou = calculate_bbox_precision_at_k_and_iou_metrics(coco_gt, coco_pred)
        eval_metrics.update({f'bbox P@{k}': m for k, m in zip([0.5, 0.6, 0.7, 0.8, 0.9], precision_at_k)})
        print(f'\n **** Bbox overall/mean IoU is {overall_iou}/{mean_iou}. **** \n')
        eval_metrics.update({'bbox overall_iou': overall_iou, 'bbox mean_iou': mean_iou})
        # mask
        precision_at_k, overall_iou, mean_iou = calculate_precision_at_k_and_iou_metrics(coco_gt, coco_pred)
        print(f'\n **** Segm overall/mean IoU is {overall_iou}/{mean_iou}. **** \n')
        eval_metrics.update({f'segm P@{k}': m for k, m in zip([0.5, 0.6, 0.7, 0.8, 0.9], precision_at_k)})
        eval_metrics.update({'segm overall_iou': overall_iou, 'segm mean_iou': mean_iou})
        print(eval_metrics)
        stats.update(eval_metrics)

    return stats


