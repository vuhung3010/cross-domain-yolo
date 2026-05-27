"""Multi-domain dataloader pairing helpers.

resolve_da_path: resolves target/aux paths against the YAML's `path:` root
  (since check_dataset only resolves train/val/test — see utils/general.py:416).

create_target_dataloader: a thin wrapper around YOLO's create_dataloader for
  unlabeled target/aux image lists. We reuse YOLO's pipeline so augmentation
  (letterbox, image scaling) matches the source pipeline exactly.
"""
from pathlib import Path
from typing import Optional


def resolve_da_path(p: str, root: Optional[str]) -> str:
    """Resolve a DA path (target/aux) against the YAML 'path:' root.

    Absolute paths are returned unchanged. Relative paths are joined with root.
    """
    pp = Path(p)
    if pp.is_absolute():
        return str(pp)
    if root is None:
        return str(pp)
    return str(Path(root) / pp)


def create_target_dataloader(images_path, imgsz, batch_size, stride, workers=4, prefix='target: '):
    """Create a dataloader for unlabeled target/aux images.

    Uses YOLO's create_dataloader so augmentation matches source. The labels it
    yields are ignored at the training loop level — we only consume images.
    """
    from utils.datasets import create_dataloader
    loader, _ = create_dataloader(
        images_path, imgsz, batch_size, stride,
        single_cls=False, hyp=None, augment=False, cache=False, rect=False,
        rank=-1, workers=workers, image_weights=False, quad=False,
        prefix=prefix, shuffle=True,
    )
    return loader
