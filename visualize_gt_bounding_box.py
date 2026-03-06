"""
Visualize ground-truth bounding boxes for a single VOC2007 image.

Usage:
    python visualize_gt_bounding_box.py --img-id 000112
    python visualize_gt_bounding_box.py --img-id 000021 --voc-root datasets/VOC2007/VOCdevkit/VOC2007 --out-dir ./output
"""

import os
import argparse
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# ── Colour palette (one colour per class, cycling if needed) ────────────────
COLOURS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9a6324", "#fffac8", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#ffffff",
]


def parse_voc_annotation(xml_path):
    """Parse a VOC XML annotation file.

    Returns a list of dicts:
        [{'name': str, 'xmin': int, 'ymin': int, 'xmax': int, 'ymax': int,
          'difficult': bool}, ...]
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        difficult = int(obj.find("difficult").text) == 1
        bb = obj.find("bndbox")
        xmin = int(float(bb.find("xmin").text))
        ymin = int(float(bb.find("ymin").text))
        xmax = int(float(bb.find("xmax").text))
        ymax = int(float(bb.find("ymax").text))
        objects.append(
            dict(name=name, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax,
                 difficult=difficult)
        )
    return objects


def visualize(img_id, voc_root, out_dir=None, show=True):
    img_path = os.path.join(voc_root, "JPEGImages", f"{img_id}.jpg")
    ann_path = os.path.join(voc_root, "Annotations", f"{img_id}.xml")

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"Annotation not found: {ann_path}")

    image = np.array(Image.open(img_path).convert("RGB"))
    objects = parse_voc_annotation(ann_path)

    # Assign a consistent colour per class name
    class_names = sorted(set(o["name"] for o in objects))
    class_colour = {name: COLOURS[i % len(COLOURS)] for i, name in enumerate(class_names)}

    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(image)
    ax.set_title(f"VOC2007 GT — {img_id}.jpg", fontsize=13)
    ax.axis("off")

    for obj in objects:
        x, y = obj["xmin"], obj["ymin"]
        w = obj["xmax"] - obj["xmin"]
        h = obj["ymax"] - obj["ymin"]
        colour = class_colour[obj["name"]]

        # Draw bounding box
        linestyle = "--" if obj["difficult"] else "-"
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=2, edgecolor=colour, facecolor="none",
            linestyle=linestyle
        )
        ax.add_patch(rect)

        # Draw label background + text
        label = obj["name"] + (" (diff)" if obj["difficult"] else "")
        ax.text(
            x, y - 4, label,
            fontsize=9, color="white", fontweight="bold",
            bbox=dict(facecolor=colour, alpha=0.8, pad=2, edgecolor="none"),
        )

    # Legend
    legend_handles = [
        patches.Patch(facecolor=class_colour[n], label=n) for n in class_names
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9,
              framealpha=0.8)

    plt.tight_layout()

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, f"{img_id}_gt_boxes.jpg")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")

    if show:
        plt.show()

    plt.close(fig)

    # Also print a summary to stdout
    print(f"\nImage : {img_id}.jpg  ({image.shape[1]}×{image.shape[0]})")
    print(f"Objects ({len(objects)}):")
    for o in objects:
        diff_str = "  [difficult]" if o["difficult"] else ""
        print(f"  {o['name']:20s}  xmin={o['xmin']:4d}  ymin={o['ymin']:4d}"
              f"  xmax={o['xmax']:4d}  ymax={o['ymax']:4d}{diff_str}")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize VOC2007 ground-truth bounding boxes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--img-id", type=str, required=True,
        help="Image ID without extension, e.g. 000112",
    )
    parser.add_argument(
        "--voc-root", type=str,
        default="datasets/VOC2007/VOCdevkit/VOC2007",
        help="Path to the VOC2007 root (containing JPEGImages/ and Annotations/)",
    )
    parser.add_argument(
        "--out-dir", type=str, default="./output",
        help="Directory to save the visualisation (set to '' to skip saving)",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Do not display the matplotlib window (useful for headless runs)",
    )
    args = parser.parse_args()

    visualize(
        img_id=args.img_id,
        voc_root=args.voc_root,
        out_dir=args.out_dir if args.out_dir else None,
        show=not args.no_show,
    )
