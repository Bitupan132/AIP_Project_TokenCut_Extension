"""
Main experiment file. Code adapted from LOST: https://github.com/valeoai/LOST
"""
import os
import argparse
import random
import pickle

import torch
import datetime
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from PIL import Image

from networks import get_model
from datasets import ImageDataset, Dataset, bbox_iou
from visualizations import visualize_img, visualize_eigvec, visualize_predictions, visualize_predictions_gt, visualize_predictions_multi
from object_discovery import ncut, ncut_recursive, ncut_multi_eigenvec, ncut_auto_k, AFFINITY_METHODS
import matplotlib.pyplot as plt
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Visualize Self-Attention maps")
    parser.add_argument(
        "--arch",
        default="vit_small",
        type=str,
        choices=[
            "vit_tiny",
            "vit_small",
            "vit_base",
            "moco_vit_small",
            "moco_vit_base",
            "mae_vit_base",
        ],
        help="Model architecture.",
    )
    parser.add_argument(
        "--patch_size", default=16, type=int, help="Patch resolution of the model."
    )

    # Use a dataset
    parser.add_argument(
        "--dataset",
        default="VOC07",
        type=str,
        choices=[None, "VOC07", "VOC12", "COCO20k"],
        help="Dataset name.",
    )
    
    parser.add_argument(
        "--save-feat-dir",
        type=str,
        default=None,
        help="if save-feat-dir is not None, only computing features and save it into save-feat-dir",
    )
    
    parser.add_argument(
        "--set",
        default="train",
        type=str,
        choices=["val", "train", "trainval", "test"],
        help="Path of the image to load.",
    )
    # Or use a single image
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="If want to apply only on one image, give file path.",
    )

    # Folder used to output visualizations and 
    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="Output directory to store predictions and visualizations."
    )

    # Evaluation setup
    parser.add_argument("--no_hard", action="store_true", help="Only used in the case of the VOC_all setup (see the paper).")
    parser.add_argument("--no_evaluation", action="store_true", help="Compute the evaluation.")
    parser.add_argument("--save_predictions", default=True, type=bool, help="Save predicted bouding boxes.")

    # Visualization
    parser.add_argument(
        "--visualize",
        type=str,
        choices=["attn", "pred", "all", None],
        default=None,
        help="Select the different type of visualizations.",
    )

    # TokenCut parameters
    parser.add_argument(
        "--which_features",
        type=str,
        default="k",
        choices=["k", "q", "v"],
        help="Which features to use",
    )
    parser.add_argument(
        "--k_patches",
        type=int,
        default=100,
        help="Number of patches with the lowest degree considered."
    )
    parser.add_argument("--resize", type=int, default=None, help="Resize input image to fix size")
    parser.add_argument("--tau", type=float, default=0.2, help="Tau for seperating the Graph.")
    parser.add_argument("--eps", type=float, default=1e-5, help="Eps for defining the Graph.")
    parser.add_argument("--no-binary-graph", action="store_true", default=False, help="Generate a binary graph where edge of the Graph will binary. Or using similarity score as edge weight.")

    # Affinity method selection
    parser.add_argument(
        "--affinity-method",
        type=str,
        default="baseline",
        choices=list(AFFINITY_METHODS),
        help=(
            "Affinity matrix construction method. "
            "'baseline': pure cosine similarity (no modification). "
            "'A': + Gaussian spatial proximity (see --spatial-weight, --spatial-sigma). "
            "'B': + graph diffusion E'=E+gamma*E^2 (see --diffusion-gamma)."
        ),
    )
    parser.add_argument(
        "--spatial-weight", type=float, default=5.0,
        help="[Method A] Weight (alpha) for the Gaussian spatial proximity kernel."
    )
    parser.add_argument(
        "--spatial-sigma", type=float, default=9.0,
        help="[Method A] Sigma in patch-grid units for the spatial Gaussian kernel."
    )
    parser.add_argument(
        "--diffusion-gamma", type=float, default=0.1,
        help="[Method B] Gamma for graph diffusion: E' = E + gamma * E^2."
    )

    # Recursive / Iterative TokenCut
    parser.add_argument(
        "--recursive", action="store_true", default=False,
        help="Use iterative (recursive) NCut for multi-object detection (Shi & Malik §3.2)."
    )
    parser.add_argument(
        "--ncut-threshold", type=float, default=0.2,
        help="[Recursive] Stop recursing when NCut energy >= this value (default 0.2)."
    )
    parser.add_argument(
        "--stability-threshold", type=float, default=0.06,
        help="[Recursive] Eigenvector histogram stability threshold (default 0.06)."
    )
    parser.add_argument(
        "--min-segment-size", type=int, default=20,
        help="[Recursive] Minimum number of patches to attempt a split (default 20)."
    )
    parser.add_argument(
        "--max-segment-ratio", type=float, default=0.5,
        help="[Recursive] Segments covering more than this fraction of all patches are "
             "treated as background and get no box (default 0.5)."
    )
    parser.add_argument(
        "--min-box-area-ratio", type=float, default=0.0,
        help="[Recursive] Discard predicted boxes whose pixel area is smaller than "
             "this fraction of the image area (default 0.0 = keep all). "
             "E.g. 0.02 removes boxes covering less than 2%% of the image."
    )

    # Multi-eigenvector spectral clustering
    parser.add_argument(
        "--multi-eigenvec", action="store_true", default=False,
        help="Use multi-eigenvector spectral clustering for multi-object detection (Ng et al. 2002)."
    )
    parser.add_argument(
        "--n-segments", type=int, default=3,
        help="[Multi-eigenvec] Number of spectral clusters / objects to detect (default 3)."
    )
    parser.add_argument(
        "--kmeans-n-init", type=int, default=10,
        help="[Multi-eigenvec / Auto-k] Number of k-means restarts (default 10)."
    )

    # Automatic-k multi-eigenvector NCut (eigengap heuristic)
    parser.add_argument(
        "--auto-k", action="store_true", default=False,
        help="Automatically determine number of objects via eigengap heuristic (Shi & Malik 2000)."
    )
    parser.add_argument(
        "--max-k", type=int, default=6,
        help="[Auto-k] Upper bound on the number of clusters to consider (default 6)."
    )

    # Use dino-seg proposed method
    parser.add_argument("--dinoseg", action="store_true", help="Apply DINO-seg baseline.")
    parser.add_argument("--dinoseg_head", type=int, default=4)

    args = parser.parse_args()

    if args.image_path is not None:
        args.save_predictions = False
        args.no_evaluation = True
        args.dataset = None

    # -------------------------------------------------------------------------------------------------------
    # Dataset

    # If an image_path is given, apply the method only to the image
    if args.image_path is not None:
        dataset = ImageDataset(args.image_path, args.resize)
    else:
        dataset = Dataset(args.dataset, args.set, args.no_hard)

    # -------------------------------------------------------------------------------------------------------
    # Model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #device = torch.device('cuda') 
    model = get_model(args.arch, args.patch_size, device)

    # -------------------------------------------------------------------------------------------------------
    # Directories
    if args.image_path is None:
        args.output_dir = os.path.join(args.output_dir, dataset.name)
    os.makedirs(args.output_dir, exist_ok=True)

    # Naming
    if args.dinoseg:
        # Experiment with the baseline DINO-seg
        if "vit" not in args.arch:
            raise ValueError("DINO-seg can only be applied to tranformer networks.")
        exp_name = f"{args.arch}-{args.patch_size}_dinoseg-head{args.dinoseg_head}"
    else:
        # Experiment with TokenCut 
        exp_name = f"TokenCut-{args.arch}"
        if "vit" in args.arch:
            exp_name += f"{args.patch_size}_{args.which_features}"

    print(f"Running TokenCut on the dataset {dataset.name} (exp: {exp_name})")

    # Visualization 
    if args.visualize:
        vis_folder = f"{args.output_dir}/{exp_name}"
        os.makedirs(vis_folder, exist_ok=True)
        
    if args.save_feat_dir is not None : 
        os.mkdir(args.save_feat_dir)

    # -------------------------------------------------------------------------------------------------------
    # Loop over images
    preds_dict = {}
    cnt = 0
    corloc = np.zeros(len(dataset.dataloader))
    
    start_time = time.time() 
    pbar = tqdm(dataset.dataloader)
    for im_id, inp in enumerate(pbar):

        # ------------ IMAGE PROCESSING -------------------------------------------
        img = inp[0]

        init_image_size = img.shape

        # Get the name of the image
        im_name = dataset.get_image_name(inp[1])
        # Pass in case of no gt boxes in the image
        if im_name is None:
            continue

        # Padding the image with zeros to fit multiple of patch-size
        size_im = (
            img.shape[0],
            int(np.ceil(img.shape[1] / args.patch_size) * args.patch_size),
            int(np.ceil(img.shape[2] / args.patch_size) * args.patch_size),
        )
        # Move to device first, then pad directly on device (avoids a CPU→GPU copy)
        img = img.to(device, non_blocking=True)
        paded = torch.zeros(size_im, device=device)
        paded[:, : img.shape[1], : img.shape[2]] = img
        img = paded
        # Size for transformers
        w_featmap = img.shape[-2] // args.patch_size
        h_featmap = img.shape[-1] // args.patch_size

        # ------------ GROUND-TRUTH -------------------------------------------
        if not args.no_evaluation:
            gt_bbxs, gt_cls = dataset.extract_gt(inp[1], im_name)

            if gt_bbxs is not None:
                # Discard images with no gt annotations
                # Happens only in the case of VOC07 and VOC12
                if gt_bbxs.shape[0] == 0 and args.no_hard:
                    continue

        # ------------ EXTRACT FEATURES -------------------------------------------
        with torch.no_grad():

            # ------------ FORWARD PASS -------------------------------------------
            if "vit"  in args.arch:
                # Store the outputs of qkv layer from the last attention layer
                feat_out = {}
                def hook_fn_forward_qkv(module, input, output):
                    feat_out["qkv"] = output
                model._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)

                # Forward pass in the model
                attentions = model.get_last_selfattention(img[None, :, :, :])

                # Scaling factor
                scales = [args.patch_size, args.patch_size]

                # Dimensions
                nb_im = attentions.shape[0]  # Batch size
                nh = attentions.shape[1]  # Number of heads
                nb_tokens = attentions.shape[2]  # Number of tokens

                # Baseline: compute DINO segmentation technique proposed in the DINO paper
                # and select the biggest component
                if args.dinoseg:
                    pred = dino_seg(attentions, (w_featmap, h_featmap), args.patch_size, head=args.dinoseg_head)
                    pred = np.asarray(pred)
                else:
                    # Extract the qkv features of the last attention layer
                    qkv = (
                        feat_out["qkv"]
                        .reshape(nb_im, nb_tokens, 3, nh, -1 // nh)
                        .permute(2, 0, 3, 1, 4)
                    )
                    q, k, v = qkv[0], qkv[1], qkv[2]
                    k = k.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
                    q = q.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
                    v = v.transpose(1, 2).reshape(nb_im, nb_tokens, -1)

                    # Modality selection
                    if args.which_features == "k":
                        #feats = k[:, 1:, :]
                        feats = k
                    elif args.which_features == "q":
                        #feats = q[:, 1:, :]
                        feats = q
                    elif args.which_features == "v":
                        #feats = v[:, 1:, :]
                        feats = v
                        
                    if args.save_feat_dir is not None : 
                        np.save(os.path.join(args.save_feat_dir, im_name.replace('.jpg', '.npy').replace('.jpeg', '.npy').replace('.png', '.npy')), feats.cpu().numpy())
                        continue

            else:
                raise ValueError("Unknown model.")

        # ------------ Apply TokenCut -------------------------------------------
        if not args.dinoseg:
            ncut_kwargs = dict(
                tau=args.tau, eps=args.eps, im_name=im_name,
                no_binary_graph=args.no_binary_graph,
                method=args.affinity_method,
                spatial_weight=args.spatial_weight,
                spatial_sigma=args.spatial_sigma,
                gamma=args.diffusion_gamma,
            )

            if args.recursive:
                preds_list, masks_list, eigvecs_list = ncut_recursive(
                    feats, [w_featmap, h_featmap], scales, init_image_size,
                    ncut_thresh=args.ncut_threshold,
                    stability_thresh=args.stability_threshold,
                    min_segment_size=args.min_segment_size,
                    max_segment_ratio=args.max_segment_ratio,
                    min_box_area_ratio=args.min_box_area_ratio,
                    **ncut_kwargs,
                )
                pred = preds_list[0]
                eigenvector = eigvecs_list[0]
            elif args.multi_eigenvec:
                preds_list, masks_list, eigvecs_list = ncut_multi_eigenvec(
                    feats, [w_featmap, h_featmap], scales, init_image_size,
                    n_segments=args.n_segments,
                    kmeans_n_init=args.kmeans_n_init,
                    **ncut_kwargs,
                )
                pred = preds_list[0]
                eigenvector = eigvecs_list[0]
            elif args.auto_k:
                k_found, preds_list, masks_list, eigvecs_list = ncut_auto_k(
                    feats, [w_featmap, h_featmap], scales, init_image_size,
                    max_k=args.max_k,
                    kmeans_n_init=args.kmeans_n_init,
                    **ncut_kwargs,
                )
                pred = preds_list[0]
                eigenvector = eigvecs_list[0]
            else:
                pred, objects, foreground, seed, bins, eigenvector = ncut(
                    feats, [w_featmap, h_featmap], scales, init_image_size,
                    **ncut_kwargs,
                )
                preds_list = [pred]

            if args.visualize in ("pred", "all") and args.no_evaluation:
                image = dataset.load_image(im_name, size_im)
                if len(preds_list) > 1:
                    visualize_predictions_multi(image, preds_list, vis_folder, im_name)
                else:
                    visualize_predictions(image, pred, vis_folder, im_name)
            if args.visualize in ("attn", "all") and args.no_evaluation:
                visualize_eigvec(eigenvector, vis_folder, im_name, [w_featmap, h_featmap], scales)

        # ------------ Visualizations -------------------------------------------
        # Save all boxes for multi-box modes so pkl evaluation matches live corloc
        if not args.dinoseg and (args.recursive or args.multi_eigenvec or args.auto_k):
            preds_dict[im_name] = preds_list
        else:
            preds_dict[im_name] = pred

        # Evaluation
        if args.no_evaluation:
            continue

        # Compare prediction to GT boxes — multi-box modes check all predictions
        if (args.recursive or args.multi_eigenvec or args.auto_k) and not args.dinoseg:
            ious = torch.cat([
                bbox_iou(torch.from_numpy(p), torch.from_numpy(gt_bbxs))
                for p in preds_list
            ])
        else:
            ious = bbox_iou(torch.from_numpy(pred), torch.from_numpy(gt_bbxs))

        if torch.any(ious >= 0.5):
            corloc[im_id] = 1
        vis_folder = f"{args.output_dir}/{exp_name}"
        os.makedirs(vis_folder, exist_ok=True)
        image = dataset.load_image(im_name)
        #visualize_predictions(image, pred, vis_folder, im_name)
        #visualize_eigvec(eigenvector, vis_folder, im_name, [w_featmap, h_featmap], scales)

        cnt += 1
        if cnt % 50 == 0:
            pbar.set_description(f"Found {int(np.sum(corloc))}/{cnt}")

    end_time = time.time()
    print(f'Time cost: {str(datetime.timedelta(milliseconds=int((end_time - start_time)*1000)))}')
    # Save predicted bounding boxes
    if args.save_predictions:
        folder = f"{args.output_dir}/{exp_name}"
        os.makedirs(folder, exist_ok=True)
        filename = os.path.join(folder, "preds.pkl")
        with open(filename, "wb") as f:
            pickle.dump(preds_dict, f)
        print("Predictions saved at %s" % filename)

    # Evaluate
    if not args.no_evaluation:
        print(f"corloc: {100*np.sum(corloc)/cnt:.2f} ({int(np.sum(corloc))}/{cnt})")
        result_file = os.path.join(folder, 'results.txt')
        with open(result_file, 'w') as f:
            f.write('corloc,%.1f,,\n'%(100*np.sum(corloc)/cnt))
        print('File saved at %s'%result_file)
