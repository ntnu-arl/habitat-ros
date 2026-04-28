from pathlib import Path

import yaml

BASE_PATH = Path("/developer/hm3d/")
SPLITS = ["train", "val"]

classes = set()
for split in SPLITS:
    split_path = BASE_PATH / split
    for scene_dir in split_path.iterdir():
        if not scene_dir.is_dir():
            continue
        scene_semantics_path = (
            scene_dir / f"{scene_dir.name.split('-')[1]}.semantic.txt"
        )
        if not scene_semantics_path.exists():
            continue
        with open(scene_semantics_path, "r") as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                label = line.strip().split(",")[2].replace('"', "")
                classes.add(label)

names = sorted(list(classes))


save_dict = {
    "label_names": [{"label": i, "name": names[i]} for i in range(len(names))],
    "surface_place_labels": [],
    "object_labels": [i for i in range(len(names))],
    "invalid_labels": [],
    "dynamic_labels": [],
    "total_semantic_labels": len(names),
}
with open(
    "/developer/ros2_hydra_ws/src/habitat-ros/config/all_semantic_labels.yaml", "w"
) as f:
    yaml.safe_dump(save_dict, f)
