# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import os

from detectron2.data import DatasetCatalog, MetadataCatalog

from .nlos import load_nlos_json

"""
This file contains functions to register a NLOS-format dataset to the DatasetCatalog.
"""

__all__ = ["register_nlos_instances", "register_nlos_panoptic_separated"]


def register_nlos_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in NLOS's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://nlosdataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.
 
    Args:
        name (str): the name that identifies a dataset, e.g. "nlos_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_nlos_json(json_file, image_root, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="nlos", **metadata
    )