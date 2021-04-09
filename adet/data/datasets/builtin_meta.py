
NLOS_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "human_01"},
    {"color": [220, 20, 60], "isthing": 1, "id": 2, "name": "human_02"},
    #{"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "human_02"},
]

def _get_nlos_instances_meta():
    thing_ids = [k["id"] for k in NLOS_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in NLOS_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 2, len(thing_ids)
    # Mapping from the incontiguous NLOS category id to an id in [0, 79]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in NLOS_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret

def _get_builtin_metadata(dataset_name):
    if dataset_name in ["nlos", "nlosGT"]:
        return _get_nlos_instances_meta()

    raise KeyError("No built-in metadata for dataset {}".format(dataset_name))
