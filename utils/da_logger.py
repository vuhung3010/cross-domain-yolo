"""Per-iteration DA loss logger -> runs/.../da_losses.csv"""
import csv
from pathlib import Path


class DALogger:
    """Append per-iter (epoch, iter, loss_det, loss_da_image, loss_triplet_img, lambda_adv, L_c)."""

    def __init__(self, save_dir: str):
        self.path = Path(save_dir) / 'da_losses.csv'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._written_header = self.path.exists()
        self._fields = ['epoch', 'iter', 'loss_det', 'loss_da_image', 'loss_triplet_img', 'lambda_adv', 'L_c']

    def log(self, **row):
        for k in self._fields:
            row.setdefault(k, '')
        write_header = not self._written_header
        with self.path.open('a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=self._fields)
            if write_header:
                w.writeheader()
                self._written_header = True
            w.writerow({k: row[k] for k in self._fields})
