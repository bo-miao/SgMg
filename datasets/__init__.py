import torch.utils.data
import torchvision

from .ytvos import build as build_ytvos
from .davis import build as build_davis
from .a2d import build as build_a2d
from .jhmdb import build as build_jhmdb
from .refexp import build as build_refexp
from .concat_dataset import build as build_joint
from .concat_dataset import build_coco as build_joint_coco
from .concat_dataset import build_joint_ytb_dvs

def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(dataset_file: str, image_set: str, args):
    if dataset_file == 'ytvos':
        print("\n **** Start to build dataset {}. **** \n".format("build_ytvos"))
        return build_ytvos(image_set, args)
    if dataset_file == 'davis':
        print("\n **** Start to build dataset {}. **** \n".format("build_davis"))
        return build_davis(image_set, args)
    if dataset_file == 'a2d':
        print("\n **** Start to build dataset {}. **** \n".format("build_a2d"))
        return build_a2d(image_set, args)
    if dataset_file == 'jhmdb':
        print("\n **** Start to build dataset {}. **** \n".format("build_jhmdb"))
        return build_jhmdb(image_set, args)
    # for pretraining
    if dataset_file == "refcoco" or dataset_file == "refcoco+" or dataset_file == "refcocog":
        print("\n **** Start to build dataset {}. **** \n".format("build_refexp"))
        return build_refexp(dataset_file, image_set, args)

    # for joint training of refcoco and ytvos, not used.
    if dataset_file == 'joint':
        print("\n **** Start to build dataset {}. **** \n".format("build_joint"))
        return build_joint(image_set, args)
    if dataset_file == 'joint_coco':
        print("\n **** Start to build dataset {}. **** \n".format("build_joint_coco"))
        return build_joint_coco(image_set, args)
    if dataset_file == 'ytvos_joint_davis':
        print("\n **** Start to build dataset {}. **** \n".format("build_joint_ytb_dvs"))
        return build_joint_ytb_dvs(image_set, args)
    raise ValueError(f'dataset {dataset_file} not supported')
