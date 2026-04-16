"""
compare_corloc.py — Compare two prediction pkl files and visualise cases where
each method wins over the other.

Usage
-----
Step 1: save baseline predictions
    python main_tokencut.py --dataset VOC07 --set trainval \
        --output_dir outputs_baseline

Step 2: save modified predictions
    python main_tokencut.py --dataset VOC07 --set trainval \
        --recursive --output_dir outputs_recursive

Step 3: run this script
    python compare_corloc.py \
        --baseline-pkl outputs_baseline/VOC07_trainval/TokenCut-vit_small16_k/preds.pkl \
        --recursive-pkl outputs_recursive/VOC07_trainval/TokenCut-vit_small16_k/preds.pkl \
        --out-dir corloc_comparison \
        --max-images 25
"""

import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import sys
sys.path.insert(0, os.path.dirname(__file__))
from datasets import Dataset, bbox_iou
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iou_any(pred_or_list, gt_bbxs):
    """Return True if ANY predicted box has IoU >= 0.5 with ANY gt box."""
    preds = pred_or_list if isinstance(pred_or_list, list) else [pred_or_list]
    for p in preds:
        p_t = torch.from_numpy(np.asarray(p, dtype=np.float32))
        if torch.any(bbox_iou(p_t, torch.from_numpy(gt_bbxs)) >= 0.5):
            return True
    return False


def _save_comparison(image, base_boxes, rec_boxes, gt_bbxs,
                     title, base_label, rec_label,
                     base_color, rec_color, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=11)

    for ax, boxes, pred_color, panel_title in [
        (axes[0], base_boxes, base_color, base_label),
        (axes[1], rec_boxes,  rec_color,  rec_label),
    ]:
        ax.imshow(image)
        for i, box in enumerate(boxes):
            x0, y0, x1, y1 = box
            rect = mpatches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                linewidth=2, edgecolor=pred_color, facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(x0 + 2, y0 + 14, str(i + 1),
                    color=pred_color, fontsize=8, fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.4, pad=1))
        for gt in gt_bbxs:
            ax.add_patch(mpatches.Rectangle(
                (gt[0], gt[1]), gt[2] - gt[0], gt[3] - gt[1],
                linewidth=2, edgecolor='dodgerblue', facecolor='none',
                linestyle='--'
            ))
        ax.set_title(panel_title)
        ax.axis('off')

    legend_elements = [
        mpatches.Patch(edgecolor=base_color,    facecolor='none', label='Baseline pred',  linewidth=2),
        mpatches.Patch(edgecolor=rec_color,     facecolor='none', label='Modified pred',  linewidth=2),
        mpatches.Patch(edgecolor='dodgerblue',  facecolor='none', label='Ground truth',   linewidth=2, linestyle='--'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=9)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-pkl', required=True,
                        help='Path to baseline preds.pkl')
    parser.add_argument('--recursive-pkl', required=True,
                        help='Path to modified method preds.pkl')
    parser.add_argument('--dataset', default='VOC07')
    parser.add_argument('--set', default='trainval')
    parser.add_argument('--out-dir', default='corloc_comparison',
                        help='Output folder for visualisations')
    parser.add_argument('--max-images', type=int, default=25,
                        help='Max images to save per category (default 25)')
    args = parser.parse_args()

    improved_dir  = os.path.join(args.out_dir, 'modified_wins')
    regressed_dir = os.path.join(args.out_dir, 'baseline_wins')
    os.makedirs(improved_dir,  exist_ok=True)
    os.makedirs(regressed_dir, exist_ok=True)

    with open(args.baseline_pkl, 'rb') as f:
        base_preds = pickle.load(f)
    with open(args.recursive_pkl, 'rb') as f:
        rec_preds = pickle.load(f)

    print(f"Baseline pkl:  {len(base_preds)} images")
    print(f"Modified pkl:  {len(rec_preds)} images")

    dataset = Dataset(args.dataset, args.set, remove_hards=False)

    improved, regressed, both_correct, both_wrong = [], [], [], []
    gt_cache = {}

    for _, inp in enumerate(dataset.dataloader):
        im_name = dataset.get_image_name(inp[1])
        if im_name is None or im_name not in base_preds or im_name not in rec_preds:
            continue

        gt_bbxs, _ = dataset.extract_gt(inp[1], im_name)
        if gt_bbxs is None or gt_bbxs.shape[0] == 0:
            continue

        base_ok = _iou_any(base_preds[im_name], gt_bbxs)
        rec_ok  = _iou_any(rec_preds[im_name],  gt_bbxs)

        if not base_ok and rec_ok:
            improved.append(im_name)
            gt_cache[im_name] = gt_bbxs
        elif base_ok and not rec_ok:
            regressed.append(im_name)
            gt_cache[im_name] = gt_bbxs
        elif base_ok and rec_ok:
            both_correct.append(im_name)
        else:
            both_wrong.append(im_name)

    total       = len(improved) + len(regressed) + len(both_correct) + len(both_wrong)
    base_correct = len(both_correct) + len(regressed)
    rec_correct  = len(both_correct) + len(improved)

    print(f"\n{'='*50}")
    print(f"Total images evaluated    : {total}")
    print(f"  Baseline CorLoc         : {100*base_correct/total:.1f}%  ({base_correct}/{total})")
    print(f"  Modified CorLoc         : {100*rec_correct/total:.1f}%  ({rec_correct}/{total})")
    print(f"\n  Modified wins (base ✗, mod ✓) : {len(improved)}")
    print(f"  Baseline wins (base ✓, mod ✗) : {len(regressed)}")
    print(f"  Both correct                  : {len(both_correct)}")
    print(f"  Both wrong                    : {len(both_wrong)}")
    print(f"{'='*50}\n")

    def _to_list(p):
        return p if isinstance(p, list) else [p]

    # ---------- Save modified-wins ----------
    print(f"Saving up to {args.max_images} 'modified wins' → {improved_dir}/")
    for idx, im_name in enumerate(improved[:args.max_images]):
        image = dataset.load_image(im_name)
        _save_comparison(
            image,
            base_boxes=_to_list(base_preds[im_name]),
            rec_boxes=_to_list(rec_preds[im_name]),
            gt_bbxs=gt_cache[im_name],
            title=f"{im_name}  —  baseline ✗   modified ✓",
            base_label="Baseline (wrong)",
            rec_label="Modified (correct)",
            base_color='red',
            rec_color='lime',
            out_path=os.path.join(improved_dir, f"{idx+1:03d}_{im_name}"),
        )
        print(f"  [{idx+1}/{min(len(improved), args.max_images)}] {im_name}")

    # ---------- Save baseline-wins ----------
    print(f"\nSaving up to {args.max_images} 'baseline wins' → {regressed_dir}/")
    for idx, im_name in enumerate(regressed[:args.max_images]):
        image = dataset.load_image(im_name)
        _save_comparison(
            image,
            base_boxes=_to_list(base_preds[im_name]),
            rec_boxes=_to_list(rec_preds[im_name]),
            gt_bbxs=gt_cache[im_name],
            title=f"{im_name}  —  baseline ✓   modified ✗",
            base_label="Baseline (correct)",
            rec_label="Modified (wrong)",
            base_color='lime',
            rec_color='red',
            out_path=os.path.join(regressed_dir, f"{idx+1:03d}_{im_name}"),
        )
        print(f"  [{idx+1}/{min(len(regressed), args.max_images)}] {im_name}")

    print(f"\nDone.")
    print(f"  Modified wins → {improved_dir}/")
    print(f"  Baseline wins → {regressed_dir}/")


if __name__ == '__main__':
    main()
