import sys
sys.path.append('./model')
import dino # model

import object_discovery as tokencut
from object_discovery import ncut_recursive_saliency
import argparse
import utils
import bilateral_solver
import os

from shutil import copyfile
import PIL.Image as Image
import cv2
import numpy as np
from tqdm import tqdm

from torchvision import transforms
import metric
import matplotlib.pyplot as plt
import skimage
import torch

# Image transformation applied to all images
ToTensor = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406),
                                                     (0.229, 0.224, 0.225)),])

def get_tokencut_binary_map(img_pth, backbone, patch_size, tau, device,
                            method="baseline", recursive=False,
                            ncut_thresh=0.2, stability_thresh=0.06,
                            min_segment_size=20, max_segment_ratio=0.5,
                            **method_kwargs):
    I = Image.open(img_pth).convert('RGB')
    # w, h are original PIL dimensions (width, height)
    I_resize, w, h, feat_w, feat_h = utils.resize_pil(I, patch_size)

    tensor = ToTensor(I_resize).unsqueeze(0).to(device)
    feat = backbone(tensor)[0]   # (D, N) — CLS already stripped by ViTFeat

    if recursive:
        bipartition, eigvec, affinity = ncut_recursive_saliency(
            feat, [feat_h, feat_w], [patch_size, patch_size], [h, w], tau,
            method=method,
            ncut_thresh=ncut_thresh,
            stability_thresh=stability_thresh,
            min_segment_size=min_segment_size,
            max_segment_ratio=max_segment_ratio,
            **method_kwargs)
    else:
        _, bipartition, eigvec, affinity = tokencut.ncut(
            feat, [feat_h, feat_w], [patch_size, patch_size], [h, w], tau,
            method=method, **method_kwargs)

    # Upsample bipartition and eigvec from feature-map resolution (feat_h, feat_w)
    # to original image resolution (h, w) — required by the bilateral solver and
    # overlay visualization. cv2.resize takes (width, height) order.
    bipartition = cv2.resize(bipartition.astype(np.float32), (w, h))
    eigvec = cv2.resize(eigvec.astype(np.float32), (w, h))

    return bipartition, eigvec, affinity

def mask_color_compose(org, mask, mask_color = [173, 216, 230]) :

    mask_fg = mask > 0.5
    rgb = np.copy(org)
    rgb[mask_fg] = (rgb[mask_fg] * 0.3 + np.array(mask_color) * 0.7).astype(np.uint8)

    return Image.fromarray(rgb)


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

## input / output dir
parser.add_argument('--out-dir', type=str, help='output directory')

parser.add_argument('--vit-arch', type=str, default='small', choices=['base', 'small'], help='which architecture')

parser.add_argument('--vit-feat', type=str, default='k', choices=['k', 'q', 'v', 'kqv'], help='which features')

parser.add_argument('--patch-size', type=int, default=16, choices=[16, 8], help='patch size')

parser.add_argument('--tau', type=float, default=0.2, help='Tau for tresholding graph')

parser.add_argument('--sigma-spatial', type=float, default=16, help='sigma spatial in the bilateral solver')

parser.add_argument('--sigma-luma', type=float, default=16, help='sigma luma in the bilateral solver')

parser.add_argument('--sigma-chroma', type=float, default=8, help='sigma chroma in the bilateral solver')

parser.add_argument('--spatial-weight', type=float, default=0.5,
                    help='[Method A] Weight (alpha) for the Gaussian spatial proximity kernel.')
parser.add_argument('--spatial-sigma', type=float, default=6.0,
                    help='[Method A] Sigma (in patch units) for the spatial Gaussian: exp(-||i-j||^2 / sigma^2).')

parser.add_argument('--affinity-method', type=str, default='baseline',
                    choices=list(tokencut.AFFINITY_METHODS),
                    help=(
                        'Affinity matrix construction method. '
                        "'baseline': pure cosine similarity. "
                        "'A': + Gaussian spatial proximity (see --spatial-weight, --spatial-sigma). "
                        "'B': + graph diffusion E'=E+gamma*E^2 (see --diffusion-gamma)."
                    ))
parser.add_argument('--diffusion-gamma', type=float, default=0.1,
                    help="[Method B] Gamma for graph diffusion: E' = E + gamma * E^2.")

# Recursive NCut
parser.add_argument('--recursive', action='store_true', default=False,
                    help='Use recursive NCut — merges all foreground leaf segments into '
                         'one saliency mask before bilateral solving.')
parser.add_argument('--ncut-threshold', type=float, default=0.2,
                    help='[Recursive] Stop recursing when NCut energy >= this value (default 0.2).')
parser.add_argument('--stability-threshold', type=float, default=0.06,
                    help='[Recursive] Eigenvector bimodality threshold (default 0.06).')
parser.add_argument('--min-segment-size', type=int, default=20,
                    help='[Recursive] Min patches to attempt a split (default 20).')
parser.add_argument('--max-segment-ratio', type=float, default=0.5,
                    help='[Recursive] Segments covering > this fraction of patches are '
                         'treated as background (default 0.5).')

parser.add_argument('--dataset', type=str, default=None, choices=['ECSSD', 'DUTS', 'DUT', None], help='which dataset?')

parser.add_argument('--nb-vis', type=int, default=100, choices=[1, 200], help='nb of visualization')

parser.add_argument('--img-path', type=str, default=None, help='single image visualization')

args = parser.parse_args()
print (args)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

## feature net

if args.vit_arch == 'base' and args.patch_size == 16:
    url = "/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
    feat_dim = 768
elif args.vit_arch == 'base' and args.patch_size == 8:
    url = "/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
    feat_dim = 768
elif args.vit_arch == 'small' and args.patch_size == 16:
    url = "/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
    feat_dim = 384
elif args.vit_arch == 'base' and args.patch_size == 8:
    url = "/dino/dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"

backbone = dino.ViTFeat(url, feat_dim, args.vit_arch, args.vit_feat, args.patch_size)
#    resume_path = './model/dino_vitbase16_pretrain.pth' if args.patch_size == 16 else './model/dino_vitbase8_pretrain.pth'

#    feat_dim = 768
#    backbone = dino.ViTFeat(resume_path, feat_dim, args.vit_arch, args.vit_feat, args.patch_size)
#
#else :
#    resume_path = './model/dino_deitsmall16_pretrain.pth' if args.patch_size == 16 else './model/dino_deitsmall8_pretrain.pth'
#    feat_dim = 384
#    backbone = dino.ViTFeat(resume_path, feat_dim, args.vit_arch, args.vit_feat, args.patch_size)


msg = 'Load {} pre-trained feature...'.format(args.vit_arch)
print (msg)
backbone.eval()
backbone.to(device)

if args.dataset == 'ECSSD' :
    args.img_dir = '../datasets/ECSSD/img'
    args.gt_dir = '../datasets/ECSSD/gt'

elif args.dataset == 'DUTS' :
    args.img_dir = '../datasets/DUTS_Test/img'
    args.gt_dir = '../datasets/DUTS_Test/gt'

elif args.dataset == 'DUT' :
    args.img_dir = '../datasets/DUT_OMRON/img'
    args.gt_dir = '../datasets/DUT_OMRON/gt'

elif args.dataset is None :
    args.gt_dir = None


print(args.dataset)

if args.out_dir is not None and not os.path.exists(args.out_dir) :
    os.mkdir(args.out_dir)

if args.img_path is not None:
    args.nb_vis = 1
    img_list = [args.img_path]
else:
    img_list = sorted(os.listdir(args.img_dir))

count_vis = 0
mask_lost = []
mask_bfs = []
gt = []
for img_name in tqdm(img_list) :
    if args.img_path is not None:
        img_pth = img_name
        img_name = img_name.split("/")[-1]
        print(img_name)
    else:
        img_pth = os.path.join(args.img_dir, img_name)
    
    bipartition, eigvec, affinity = get_tokencut_binary_map(
        img_pth, backbone, args.patch_size, args.tau, device,
        method=args.affinity_method,
        recursive=args.recursive,
        ncut_thresh=args.ncut_threshold,
        stability_thresh=args.stability_threshold,
        min_segment_size=args.min_segment_size,
        max_segment_ratio=args.max_segment_ratio,
        spatial_weight=args.spatial_weight, spatial_sigma=args.spatial_sigma,
        gamma=args.diffusion_gamma)
    mask_lost.append(bipartition)

    output_solver, binary_solver = bilateral_solver.bilateral_solver_output(img_pth, bipartition, sigma_spatial = args.sigma_spatial, sigma_luma = args.sigma_luma, sigma_chroma = args.sigma_chroma)
    mask1 = torch.from_numpy(bipartition).cpu()
    mask2 = torch.from_numpy(binary_solver).cpu()
    if metric.IoU(mask1, mask2) < 0.5:
        binary_solver = binary_solver * -1
    mask_bfs.append(output_solver)

    if args.gt_dir is not None :
        mask_gt = np.array(Image.open(os.path.join(args.gt_dir, img_name.replace('.jpg', '.png'))).convert('L'))
        gt.append(mask_gt)

    if count_vis != args.nb_vis :
        print(f'args.out_dir: {args.out_dir}, img_name: {img_name}')
        out_name = os.path.join(args.out_dir, img_name)
        out_lost = os.path.join(args.out_dir, img_name.replace('.jpg', '_tokencut.jpg'))
        out_bfs = os.path.join(args.out_dir, img_name.replace('.jpg', '_tokencut_bfs.jpg'))
        out_affinity = os.path.join(args.out_dir, img_name.replace('.jpg', '_affinity.png'))
        out_eigvec = os.path.join(args.out_dir, img_name.replace('.jpg', '_eigen_attention.png'))
        out_eigvec_overlay = os.path.join(args.out_dir, img_name.replace('.jpg', '_eigen_attention_overlay.png'))

        copyfile(img_pth, out_name)
        org = np.array(Image.open(img_pth).convert('RGB'))

        # --- Eigen-attention: standalone heatmap ---
        # Use diverging colormap (RdBu_r) since eigvec has positive & negative values
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(eigvec, cmap='RdBu_r', interpolation='bilinear')
        plt.colorbar(im, ax=ax, label='Eigenvector value')
        ax.set_title('Eigen-Attention (2nd smallest eigenvector)')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(out_eigvec, dpi=150)
        plt.close(fig)

        # --- Eigen-attention: overlay on original image ---
        eigvec_norm = (eigvec - eigvec.min()) / (eigvec.max() - eigvec.min() + 1e-8)  # [0,1]
        cmap_fn = plt.get_cmap('RdBu_r')
        eigvec_color = (cmap_fn(eigvec_norm)[:, :, :3] * 255).astype(np.uint8)  # H x W x 3
        overlay = (org * 0.5 + eigvec_color * 0.5).astype(np.uint8)
        Image.fromarray(overlay).save(out_eigvec_overlay)

        mask_color_compose(org, bipartition).save(out_lost)
        mask_color_compose(org, binary_solver).save(out_bfs)

        # Visualize affinity matrix
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(affinity, cmap='viridis', interpolation='nearest')
        plt.colorbar(im, ax=ax, label='Cosine Similarity')
        ax.set_title('Affinity Matrix (patch × patch)')
        ax.set_xlabel('Patch index')
        ax.set_ylabel('Patch index')
        plt.tight_layout()
        plt.savefig(out_affinity, dpi=150)
        plt.close(fig)
        if args.gt_dir is not None :
            out_gt = os.path.join(args.out_dir, img_name.replace('.jpg', '_gt.jpg'))
            mask_color_compose(org, mask_gt).save(out_gt)


        count_vis += 1
    else :
        continue

if args.gt_dir is not None and args.img_path is None:
    print ('TokenCut evaluation:')
    print (metric.metrics(mask_lost, gt))
    print ('\n')

    print ('TokenCut + bilateral solver evaluation:')
    print (metric.metrics(mask_bfs, gt))
    print ('\n')
    print ('\n')