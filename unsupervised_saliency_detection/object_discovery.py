# """
# Main functions for applying Normalized Cut.
# Code adapted from LOST: https://github.com/valeoai/LOST
# """

# import torch
# import torch.nn.functional as F
# import numpy as np
# from scipy.linalg import eigh
# from scipy import ndimage

# def ncut(feats, dims, scales, init_image_size, tau = 0, eps=1e-5, im_name='', no_binary_graph=False):
#     """
#     Implementation of NCut Method.
#     Inputs
#       feats: the pixel/patche features of an image
#       dims: dimension of the map from which the features are used
#       scales: from image to map scale
#       init_image_size: size of the image
#       tau: thresold for graph construction
#       eps: graph edge weight
#       im_name: image_name
#       no_binary_graph: ablation study for using similarity score as graph edge weight
#     """
#     cls_token = feats[0,0:1,:].cpu().numpy() 

#     feats = feats[0,1:,:]
#     feats = F.normalize(feats, p=2)
#     A = (feats @ feats.transpose(1,0)) 
#     A = A.cpu().numpy()
#     if no_binary_graph:
#         A[A<tau] = eps
#     else:
#         A = A > tau
#         A = np.where(A.astype(float) == 0, eps, A)
#     d_i = np.su m(A, axis=1)
#     D = np.diag(d_i)
  
#     # Print second and third smallest eigenvector 
#     _, eigenvectors = eigh(D-A, D, subset_by_index=[1,2])
#     eigenvec = np.copy(eigenvectors[:, 0])

#     # Using average point to compute bipartition 
#     second_smallest_vec = eigenvectors[:, 0]
#     avg = np.sum(second_smallest_vec) / len(second_smallest_vec)
#     bipartition = second_smallest_vec > avg
    
#     seed = np.argmax(np.abs(second_smallest_vec))

#     if bipartition[seed] != 1:
#         eigenvec = eigenvec * -1
#         bipartition = np.logical_not(bipartition)
#     bipartition = bipartition.reshape(dims).astype(float)

#     # predict BBox
#     pred, _, objects,cc = detect_box(bipartition, seed, dims, scales=scales, initial_im_size=init_image_size[1:]) ## We only extract the principal object BBox
#     mask = np.zeros(dims)
#     mask[cc[0],cc[1]] = 1

#     return np.asarray(pred), objects, mask, seed, None, eigenvec.reshape(dims)

# def detect_box(bipartition, seed,  dims, initial_im_size=None, scales=None, principle_object=True):
#     """
#     Extract a box corresponding to the seed patch. Among connected components extract from the affinity matrix, select the one corresponding to the seed patch.
#     """
#     w_featmap, h_featmap = dims
#     objects, num_objects = ndimage.label(bipartition) 
#     cc = objects[np.unravel_index(seed, dims)]
    

#     if principle_object:
#         mask = np.where(objects == cc)
#        # Add +1 because excluded max
#         ymin, ymax = min(mask[0]), max(mask[0]) + 1
#         xmin, xmax = min(mask[1]), max(mask[1]) + 1
#         # Rescale to image size
#         r_xmin, r_xmax = scales[1] * xmin, scales[1] * xmax
#         r_ymin, r_ymax = scales[0] * ymin, scales[0] * ymax
#         pred = [r_xmin, r_ymin, r_xmax, r_ymax]
         
#         # Check not out of image size (used when padding)
#         if initial_im_size:
#             pred[2] = min(pred[2], initial_im_size[1])
#             pred[3] = min(pred[3], initial_im_size[0])
        
#         # Coordinate predictions for the feature space
#         # Axis different then in image space
#         pred_feats = [ymin, xmin, ymax, xmax]

#         return pred, pred_feats, objects, mask
#     else:
#         raise NotImplementedError


import torch
import torch.nn.functional as F
import numpy as np
#from scipy.linalg.decomp import eig
import scipy
from scipy.linalg import eigh
from scipy import ndimage
from sklearn.cluster import KMeans

def find_best_ncut_threshold(eigenvec, W, d_i, n_thresholds=40, return_ncut=False):
    """
    Find the threshold that minimises the normalised-cut energy (Shi & Malik 2000).

    NCut(A,B) = cut(A,B)/assoc(A,V)  +  cut(A,B)/assoc(B,V)

    where
      cut(A,B)   = sum of W[i,j] for i in A, j in B
      assoc(A,V) = sum of W[i,j] for i in A, all j  = sum of d_i[i] for i in A

    We sweep `n_thresholds` candidate values uniformly between
    min(eigenvec) and max(eigenvec) and pick the cheapest cut.

    Args:
        return_ncut: if True, return (best_thresh, best_energy) instead of just best_thresh.
    """
    lo, hi = eigenvec.min(), eigenvec.max()
    thresholds = np.linspace(lo, hi, n_thresholds + 2)[1:-1]  # exclude endpoints

    best_thresh = thresholds[0]
    best_energy = np.inf

    assoc_total = d_i.sum()  # == assoc(V, V)

    for t in thresholds:
        mask_A = eigenvec >= t   # boolean, shape (N,)
        mask_B = ~mask_A

        assoc_A = d_i[mask_A].sum()
        assoc_B = assoc_total - assoc_A

        if assoc_A == 0 or assoc_B == 0:
            continue  # degenerate cut – skip

        # cut(A,B) = sum_{i in A, j in B} W[i,j]
        # Efficient: cut = assoc_A - sum_{i in A, j in A} W[i,j]
        W_AA = W[np.ix_(mask_A, mask_A)].sum()
        cut_AB = assoc_A - W_AA

        energy = cut_AB / assoc_A + cut_AB / assoc_B

        if energy < best_energy:
            best_energy = energy
            best_thresh = t

    if return_ncut:
        return best_thresh, best_energy
    return best_thresh


# ---------------------------------------------------------------------------
# Affinity-matrix construction methods
# ---------------------------------------------------------------------------

def _apply_affinity_baseline(A, dims, **kwargs):
    """Baseline: pure cosine similarity — no modification."""
    return A


def _apply_affinity_method_A(A, dims, spatial_weight=1.0, spatial_sigma=3.0, **kwargs):
    """Method A - Spatial continuity constraint (proposal sec. 2a).

    Blends cosine similarity with a Gaussian spatial proximity kernel so that
    nearby patches stay connected when appearance differs due to partial occlusion:
        A' = A + spatial_weight * exp(-||i-j||^2 / spatial_sigma^2)

    Args:
        spatial_weight: scalar alpha blending the spatial kernel (default 5.0).
        spatial_sigma:  sigma in patch-grid units for the Gaussian (default 9.0).
    """
    feat_h, feat_w = dims
    device = A.device
    rows = torch.arange(feat_h, device=device).repeat_interleave(feat_w)
    cols = torch.arange(feat_w, device=device).repeat(feat_h)
    dr = (rows.unsqueeze(1) - rows.unsqueeze(0)).float()
    dc = (cols.unsqueeze(1) - cols.unsqueeze(0)).float()
    dist2 = dr ** 2 + dc ** 2
    spatial_kernel = torch.exp(-dist2 / (spatial_sigma ** 2))
    return A + spatial_weight * spatial_kernel


def _apply_affinity_method_B(A, dims, gamma=0.1, **kwargs):
    """Method B - Multi-hop graph diffusion (proposal sec. 2b).

    Reconnects object parts separated by an occluder through two-hop paths:
        E' = E + gamma * E^2

    Args:
        gamma: weight of the second-order diffusion term (default 0.1).
    """
    return A + gamma * (A @ A)


# ---------------------------------------------------------------------------
# Registry — add new methods here to make them available via --affinity-method.
# Keys become the valid argument values.  Uncomment / extend for methods C, D.
# ---------------------------------------------------------------------------
AFFINITY_METHODS = {
    "baseline": _apply_affinity_baseline,
    "A":        _apply_affinity_method_A,
    "B":        _apply_affinity_method_B,
    # "C": _apply_affinity_method_C,
    # "D": _apply_affinity_method_D,
}


def ncut(feats, dims, scales, init_image_size, tau=0, eps=1e-5,
         im_name='', no_binary_graph=False,
         method="baseline", **method_kwargs):
    """
    NCut for saliency detection.

    Args:
        feats: patch features from ViTFeat — shape (D, N), CLS already stripped.
               D = feature dimension, N = feat_h * feat_w patch count.
        dims:  [feat_h, feat_w]
        scales, init_image_size: unused (kept for API compatibility)
        tau, eps:  graph construction thresholds
        method:    affinity method key (see AFFINITY_METHODS)
        **method_kwargs: forwarded to the affinity method

    Returns:
        seed        — int, index of the foreground seed patch
        bipartition — (feat_h, feat_w) float array, 1=foreground 0=background
        eigvec      — (feat_h, feat_w) float array, second smallest eigenvector
        affinity    — (N, N) numpy array, the thresholded affinity matrix
    """
    device = feats.device

    # feats: (D, N) — already column-normalised to unit length by ViTFeat
    feats = F.normalize(feats, p=2, dim=0)      # L2-normalise each column
    A = feats.transpose(0, 1) @ feats           # (N, N) cosine similarity

    if method not in AFFINITY_METHODS:
        raise ValueError(f"Unknown affinity method '{method}'. Choose from: {list(AFFINITY_METHODS)}")
    A = AFFINITY_METHODS[method](A, dims, **method_kwargs)

    if no_binary_graph:
        A = torch.where(A < tau, torch.tensor(eps, device=device, dtype=A.dtype), A)
    else:
        A = (A > tau).to(A.dtype)
        A = torch.where(A == 0, torch.tensor(eps, device=device, dtype=A.dtype), A)

    d_i = A.sum(dim=1)

    D_inv_sqrt = torch.diag(1.0 / d_i.sqrt())
    A_sym = D_inv_sqrt @ A @ D_inv_sqrt
    L_sym = torch.eye(A_sym.shape[0], device=device, dtype=A_sym.dtype) - A_sym
    L_sym = (L_sym + L_sym.T) * 0.5
    _, U = torch.linalg.eigh(L_sym)
    eigvec = (D_inv_sqrt @ U[:, 1]).cpu().numpy()   # 2nd smallest eigenvector

    A_np = A.cpu().numpy()

    avg = eigvec.mean()
    bipartition = eigvec > avg

    seed = int(np.argmax(np.abs(eigvec)))

    if not bipartition[seed]:
        eigvec = eigvec * -1
        bipartition = ~bipartition

    feat_h, feat_w = dims
    bipartition_2d = bipartition.reshape(feat_h, feat_w).astype(float)
    eigvec_2d      = eigvec.reshape(feat_h, feat_w)

    return seed, bipartition_2d, eigvec_2d, A_np

def detect_box(bipartition, seed,  dims, initial_im_size=None, scales=None, principle_object=True):
    """
    Extract a box corresponding to the seed patch. Among connected components extract from the affinity matrix, select the one corresponding to the seed patch.
    """
    w_featmap, h_featmap = dims
    objects, num_objects = ndimage.label(bipartition)
    cc = objects[np.unravel_index(seed, dims)]


    if principle_object:
        mask = np.where(objects == cc)
       # Add +1 because excluded max
        ymin, ymax = min(mask[0]), max(mask[0]) + 1
        xmin, xmax = min(mask[1]), max(mask[1]) + 1
        # Rescale to image size
        r_xmin, r_xmax = scales[1] * xmin, scales[1] * xmax
        r_ymin, r_ymax = scales[0] * ymin, scales[0] * ymax
        pred = [r_xmin, r_ymin, r_xmax, r_ymax]

        # Check not out of image size (used when padding)
        if initial_im_size:
            pred[2] = min(pred[2], initial_im_size[1])
            pred[3] = min(pred[3], initial_im_size[0])

        # Coordinate predictions for the feature space
        # Axis different then in image space
        pred_feats = [ymin, xmin, ymax, xmax]

        return pred, pred_feats, objects, mask
    else:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Recursive / Iterative NCut (Shi & Malik 2000, §3.2)
# ---------------------------------------------------------------------------

def _is_eigenvec_stable(eigenvec, n_bins=10, stability_thresh=0.06):
    """
    Stability check from Shi & Malik §3: the second smallest eigenvector of a
    well-separated subgraph should have a clear bimodal distribution.

    We build a histogram with `n_bins` bins and check that the ratio of
        min_non_empty_bin_count / max_non_empty_bin_count
    is above `stability_thresh`.  A low ratio means the distribution is
    strongly bimodal → the bipartition is stable → we should recurse.
    A high ratio (all bins roughly equal) means no clear cut → stop.

    Returns True when the cut is considered stable (recurse further).
    """
    counts, _ = np.histogram(eigenvec, bins=n_bins)
    non_empty = counts[counts > 0]
    if len(non_empty) < 2:
        return False
    ratio = non_empty.min() / non_empty.max()
    return ratio < stability_thresh


def _bipartition_subgraph(A_sub_np, device, n_thresholds=50):
    """
    Run a single NCut bipartition on a sub-graph given by its adjacency
    matrix `A_sub_np` (already thresholded, numpy float32).

    Returns (bipartition_mask, ncut_value, eigenvec_np) on success,
    or None when the subgraph is too degenerate to partition.

    The eigendecomposition is run on `device` (GPU when available).
    """
    n = A_sub_np.shape[0]
    A = torch.from_numpy(A_sub_np).to(device)

    d_i = A.sum(dim=1)
    if (d_i == 0).any():
        return None   # isolated nodes — degenerate

    # Normalised Laplacian  L_sym = I - D^{-1/2} A D^{-1/2}
    D_inv_sqrt = torch.diag(1.0 / d_i.sqrt())
    A_sym = D_inv_sqrt @ A @ D_inv_sqrt
    L_sym = torch.eye(n, device=device, dtype=A_sym.dtype) - A_sym
    L_sym = (L_sym + L_sym.T) * 0.5   # enforce symmetry

    try:
        _, U = torch.linalg.eigh(L_sym)
    except RuntimeError:
        return None

    # Back-transform to generalised eigenvector
    v = (D_inv_sqrt @ U[:, 1]).cpu().numpy()   # 2nd smallest

    d_i_np = d_i.cpu().numpy()
    A_np   = A.cpu().numpy()

    best_t, ncut_val = find_best_ncut_threshold(v, A_np, d_i_np,
                                                n_thresholds=n_thresholds,
                                                return_ncut=True)
    bipartition = v >= best_t

    # Flip so the larger-magnitude side is foreground (seed convention)
    seed_local = int(np.argmax(np.abs(v)))
    if not bipartition[seed_local]:
        bipartition = ~bipartition

    return bipartition, ncut_val, v


def _add_leaf_box(node_indices, eigvec_vals, dims, scales, init_image_size_hw, results):
    """Build a bounding box for a leaf segment and append it to results."""
    feat_h, feat_w = dims
    mask_2d = np.zeros((feat_h, feat_w), dtype=float)
    rows = node_indices // feat_w
    cols = node_indices % feat_w
    mask_2d[rows, cols] = 1.0

    eigvec_2d = np.zeros((feat_h, feat_w), dtype=np.float32)
    eigvec_2d[rows, cols] = eigvec_vals

    seed_local  = int(np.argmax(np.abs(eigvec_vals)))
    seed_global = int(node_indices[seed_local])

    try:
        pred, _, _, _ = detect_box(
            mask_2d, seed_global, dims,
            scales=scales,
            initial_im_size=init_image_size_hw
        )
        results.append((np.asarray(pred), mask_2d, eigvec_2d))
    except Exception:
        pass


def _recursive_ncut_helper(A_full_np, node_indices, dims, scales,
                            init_image_size_hw, device,
                            ncut_thresh, stability_thresh, min_segment_size,
                            total_N, max_segment_ratio, n_thresholds, results):
    """
    Recursive worker for iterative NCut.

    Boxes are added ONLY at leaf nodes (segments that cannot be split further).
    Segments covering more than max_segment_ratio of all patches are silently
    dropped — they correspond to whole-image background boxes.
    """
    n = len(node_indices)
    if n < min_segment_size:
        return   # too small — discard

    # Background suppression: skip leaf-box if this segment is too large
    is_background = (n / total_N) > max_segment_ratio

    A_sub_np = A_full_np[np.ix_(node_indices, node_indices)].astype(np.float32)
    out = _bipartition_subgraph(A_sub_np, device, n_thresholds=n_thresholds)

    if out is None:
        if not is_background:
            _add_leaf_box(node_indices, np.ones(n, dtype=np.float32),
                          dims, scales, init_image_size_hw, results)
        return

    bipartition_local, ncut_val, eigvec_local = out

    # Stopping criteria → leaf node
    if (ncut_val >= ncut_thresh or
            not _is_eigenvec_stable(eigvec_local, stability_thresh=stability_thresh)):
        if not is_background:
            _add_leaf_box(node_indices, eigvec_local,
                          dims, scales, init_image_size_hw, results)
        return

    # Valid split — recurse on both halves, no box at this level
    idx_A = node_indices[bipartition_local]
    idx_B = node_indices[~bipartition_local]

    for idx_half in (idx_A, idx_B):
        _recursive_ncut_helper(A_full_np, idx_half, dims, scales,
                               init_image_size_hw, device,
                               ncut_thresh, stability_thresh, min_segment_size,
                               total_N, max_segment_ratio, n_thresholds, results)


def ncut_recursive(feats, dims, scales, init_image_size, tau=0, eps=1e-5,
                   im_name='', no_binary_graph=False,
                   method="baseline",
                   ncut_thresh=0.2, stability_thresh=0.06,
                   min_segment_size=20, max_segment_ratio=0.5,
                   n_thresholds=50,
                   **method_kwargs):
    """
    Iterative / recursive TokenCut (Shi & Malik 2000 §3.2).

    Performs a top-down recursive bipartition of the patch graph.  Each
    sub-graph is cut until the NCut energy exceeds `ncut_thresh`, the
    eigenvector is not stably bimodal, or the segment is too small.

    Args:
        feats:              Raw ViT features (1, N_tokens, D) — batch + CLS included.
        dims:               [feat_h, feat_w] of the feature map.
        scales:             [scale_h, scale_w] patch→image pixel scaling.
        init_image_size:    Shape of the original (possibly padded) input image.
        tau, eps:           Graph construction thresholds (same as ncut()).
        no_binary_graph:    Use soft edge weights instead of binary.
        method:             Affinity method key (see AFFINITY_METHODS).
        ncut_thresh:        Maximum NCut energy to continue recursion (default 0.2).
        stability_thresh:   Eigenvector histogram stability threshold (default 0.06).
        min_segment_size:   Minimum number of patches to attempt a split (default 20).
        n_thresholds:       Threshold sweep resolution (default 50).
        **method_kwargs:    Hyperparameters forwarded to the affinity method.

    Returns:
        preds   — list of np.ndarray bboxes  [x_min, y_min, x_max, y_max]
        masks   — list of 2-D float arrays   (feat_h × feat_w)
        eigvecs — list of 2-D float arrays   (feat_h × feat_w)
    """
    # ---------- Build full affinity matrix (same pipeline as ncut()) ----------
    feats_t = feats[0, 1:, :]           # (N, D) — strip batch dim & CLS token
    feats_t = feats_t.transpose(0, 1)   # (D, N)
    device  = feats_t.device

    feats_t = F.normalize(feats_t, p=2, dim=0)
    A = feats_t.transpose(0, 1) @ feats_t   # (N, N) cosine similarity

    if method not in AFFINITY_METHODS:
        raise ValueError(f"Unknown affinity method '{method}'. "
                         f"Choose from: {list(AFFINITY_METHODS)}")
    A = AFFINITY_METHODS[method](A, dims, **method_kwargs)

    if no_binary_graph:
        A = torch.where(A < tau, torch.tensor(eps, device=device, dtype=A.dtype), A)
    else:
        A = (A > tau).to(A.dtype)
        A = torch.where(A == 0, torch.tensor(eps, device=device, dtype=A.dtype), A)

    A_np = A.cpu().numpy().astype(np.float32)
    N = A_np.shape[0]

    # ---------- Recursive bipartition ----------
    results = []   # list of (pred_bbox, mask_2d, eigvec_2d)
    all_indices = np.arange(N, dtype=np.int64)
    init_image_size_hw = init_image_size[1:]   # (H, W) — strip channel dim

    _recursive_ncut_helper(
        A_np, all_indices, dims, scales, init_image_size_hw, device,
        ncut_thresh, stability_thresh, min_segment_size,
        N, max_segment_ratio, n_thresholds,
        results
    )

    if not results:
        # Fallback: single ncut so we never return empty-handed
        pred, objects, mask, seed, _, eigvec = ncut(
            feats, dims, scales, init_image_size, tau, eps,
            im_name=im_name, no_binary_graph=no_binary_graph,
            method=method, **method_kwargs
        )
        return [pred], [mask], [eigvec]

    preds, masks, eigvecs = zip(*results)
    return list(preds), list(masks), list(eigvecs)


# ---------------------------------------------------------------------------
# Recursive NCut for saliency detection
# ---------------------------------------------------------------------------

def _collect_leaf_indices_saliency(A_full_np, node_indices, device,
                                    ncut_thresh, stability_thresh, min_segment_size,
                                    total_N, max_segment_ratio, n_thresholds,
                                    foreground_indices):
    """
    Recursive worker that collects patch indices belonging to foreground
    leaf segments.  Background leaves (covering > max_segment_ratio of all
    patches) are silently discarded.  The collected indices are later merged
    into a single binary saliency mask.
    """
    n = len(node_indices)
    if n < min_segment_size:
        return

    is_background = (n / total_N) > max_segment_ratio

    A_sub_np = A_full_np[np.ix_(node_indices, node_indices)].astype(np.float32)
    out = _bipartition_subgraph(A_sub_np, device, n_thresholds=n_thresholds)

    if out is None:
        if not is_background:
            foreground_indices.extend(node_indices.tolist())
        return

    bipartition_local, ncut_val, eigvec_local = out

    if (ncut_val >= ncut_thresh or
            not _is_eigenvec_stable(eigvec_local, stability_thresh=stability_thresh)):
        if not is_background:
            foreground_indices.extend(node_indices.tolist())
        return

    idx_A = node_indices[bipartition_local]
    idx_B = node_indices[~bipartition_local]

    for idx_half in (idx_A, idx_B):
        _collect_leaf_indices_saliency(
            A_full_np, idx_half, device,
            ncut_thresh, stability_thresh, min_segment_size,
            total_N, max_segment_ratio, n_thresholds,
            foreground_indices
        )


def ncut_recursive_saliency(feats, dims, scales, init_image_size, tau=0, eps=1e-5,
                             im_name='', no_binary_graph=False,
                             method="baseline",
                             ncut_thresh=0.2, stability_thresh=0.06,
                             min_segment_size=20, max_segment_ratio=0.5,
                             n_thresholds=50, **method_kwargs):
    """
    Recursive NCut adapted for saliency detection.

    Instead of producing bounding boxes, all non-background leaf segments are
    merged into a single binary foreground mask, which is then passed to the
    bilateral solver for smoothing — exactly the same pipeline as the single-cut
    baseline, just with a richer foreground mask.

    Args:
        feats:            (D, N) patch features from ViTFeat — CLS already stripped.
        dims:             [feat_h, feat_w]
        ncut_thresh:      Stop recursing when NCut energy >= this value (default 0.2).
        stability_thresh: Eigenvector bimodality threshold (default 0.06).
        min_segment_size: Min patches to attempt a split (default 20).
        max_segment_ratio: Segments > this fraction of all patches = background (default 0.5).
        **method_kwargs:  Forwarded to the affinity method.

    Returns:
        bipartition — (feat_h, feat_w) float mask, 1=foreground 0=background
        eigvec      — (feat_h, feat_w) second-smallest eigenvector of the first split
        affinity    — (N, N) numpy affinity matrix
    """
    feat_h, feat_w = dims
    device = feats.device

    feats_n = F.normalize(feats, p=2, dim=0)
    A = feats_n.transpose(0, 1) @ feats_n   # (N, N) cosine similarity

    if method not in AFFINITY_METHODS:
        raise ValueError(f"Unknown affinity method '{method}'. "
                         f"Choose from: {list(AFFINITY_METHODS)}")
    A = AFFINITY_METHODS[method](A, dims, **method_kwargs)

    if no_binary_graph:
        A = torch.where(A < tau, torch.tensor(eps, device=device, dtype=A.dtype), A)
    else:
        A = (A > tau).to(A.dtype)
        A = torch.where(A == 0, torch.tensor(eps, device=device, dtype=A.dtype), A)

    A_np = A.cpu().numpy().astype(np.float32)
    N = A_np.shape[0]

    # Compute first-split eigenvector for visualisation
    d_i = A.sum(dim=1)
    D_inv_sqrt = torch.diag(1.0 / d_i.sqrt())
    A_sym = D_inv_sqrt @ A @ D_inv_sqrt
    L_sym = torch.eye(N, device=device, dtype=A_sym.dtype) - A_sym
    L_sym = (L_sym + L_sym.T) * 0.5
    _, U = torch.linalg.eigh(L_sym)
    eigvec_2d = (D_inv_sqrt @ U[:, 1]).cpu().numpy().reshape(feat_h, feat_w)

    # Collect foreground leaf patch indices
    foreground_indices = []
    all_indices = np.arange(N, dtype=np.int64)

    _collect_leaf_indices_saliency(
        A_np, all_indices, device,
        ncut_thresh, stability_thresh, min_segment_size,
        N, max_segment_ratio, n_thresholds,
        foreground_indices
    )

    if not foreground_indices:
        # Fallback: single-cut baseline
        _, bipartition, eigvec_2d, A_np = ncut(
            feats, dims, scales, init_image_size, tau, eps,
            im_name=im_name, no_binary_graph=no_binary_graph,
            method=method, **method_kwargs
        )
        return bipartition, eigvec_2d, A_np

    # Build combined foreground mask
    fg_arr = np.array(foreground_indices, dtype=np.int64)
    mask_2d = np.zeros((feat_h, feat_w), dtype=float)
    mask_2d[fg_arr // feat_w, fg_arr % feat_w] = 1.0

    return mask_2d, eigvec_2d, A_np


# ---------------------------------------------------------------------------
# Multi-eigenvector spectral clustering (Ng, Jordan & Weiss 2002)
# ---------------------------------------------------------------------------

def ncut_multi_eigenvec(feats, dims, scales, init_image_size, tau=0, eps=1e-5,
                        im_name='', no_binary_graph=False,
                        method="baseline", n_segments=3,
                        kmeans_n_init=10, kmeans_max_iter=300,
                        **method_kwargs):
    """
    Multi-eigenvector spectral clustering for multi-object detection.

    Instead of a single bipartition on the 2nd eigenvector, this method:
      1. Computes the `n_segments` smallest non-trivial eigenvectors of the
         normalised Laplacian (indices 1 … n_segments).
      2. Stacks them into an (N, n_segments) spectral embedding.
      3. Row-normalises the embedding to unit length (Ng et al. 2002 step 3).
      4. Runs k-means with k = n_segments to assign each patch to a cluster.
      5. Converts each cluster to a bounding box via detect_box.

    Args:
        feats:            Raw ViT features (1, N_tokens, D) — batch + CLS included.
        dims:             [feat_h, feat_w] of the feature map.
        scales:           [scale_h, scale_w] patch→image pixel scaling.
        init_image_size:  Shape of the original (possibly padded) input image.
        tau, eps:         Graph construction thresholds.
        no_binary_graph:  Use soft edge weights instead of binary.
        method:           Affinity method key (see AFFINITY_METHODS).
        n_segments:       Number of clusters / objects to detect (default 3).
        kmeans_n_init:    K-means restarts (default 10).
        kmeans_max_iter:  K-means max iterations (default 300).
        **method_kwargs:  Hyperparameters forwarded to the affinity method.

    Returns:
        preds   — list of np.ndarray bboxes  [x_min, y_min, x_max, y_max]
        masks   — list of 2-D float arrays   (feat_h × feat_w)
        eigvecs — list of 2-D float arrays   (feat_h × feat_w) — 1st embedding dim
    """
    # ---------- Build full affinity matrix (same pipeline as ncut()) ----------
    feats_t = feats[0, 1:, :]           # (N, D) — strip batch dim & CLS token
    feats_t = feats_t.transpose(0, 1)   # (D, N)
    device  = feats_t.device

    feats_t = F.normalize(feats_t, p=2, dim=0)
    A = feats_t.transpose(0, 1) @ feats_t   # (N, N) cosine similarity

    if method not in AFFINITY_METHODS:
        raise ValueError(f"Unknown affinity method '{method}'. "
                         f"Choose from: {list(AFFINITY_METHODS)}")
    A = AFFINITY_METHODS[method](A, dims, **method_kwargs)

    if no_binary_graph:
        A = torch.where(A < tau, torch.tensor(eps, device=device, dtype=A.dtype), A)
    else:
        A = (A > tau).to(A.dtype)
        A = torch.where(A == 0, torch.tensor(eps, device=device, dtype=A.dtype), A)

    d_i = A.sum(dim=1)

    # ---------- Normalised Laplacian eigendecomposition on GPU ----------
    D_inv_sqrt = torch.diag(1.0 / d_i.sqrt())
    A_sym = D_inv_sqrt @ A @ D_inv_sqrt
    L_sym = torch.eye(A_sym.shape[0], device=device, dtype=A_sym.dtype) - A_sym
    L_sym = (L_sym + L_sym.T) * 0.5

    _, U = torch.linalg.eigh(L_sym)   # eigenvalues ascending

    # Back-transform: V[:,i] = D^{-1/2} u_i  for eigenvectors 1..n_segments
    n_vecs = min(n_segments, U.shape[1] - 1)  # guard against tiny graphs
    V = (D_inv_sqrt @ U[:, 1:n_vecs + 1]).cpu().numpy()   # (N, n_vecs)

    # ---------- Row-normalise (Ng et al. 2002 step 3) ----------
    row_norms = np.linalg.norm(V, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0
    V_norm = V / row_norms

    # ---------- K-means in spectral space ----------
    kmeans = KMeans(n_clusters=n_segments, n_init=kmeans_n_init,
                    max_iter=kmeans_max_iter, random_state=0)
    labels = kmeans.fit_predict(V_norm)   # (N,) int cluster ids

    # ---------- Map each cluster → bounding box ----------
    feat_h, feat_w = dims
    preds, masks, eigvecs = [], [], []

    # Use the 1st spectral dimension for eigenvec visualisation
    eigvec_full = V[:, 0].reshape(feat_h, feat_w)

    for cluster_id in range(n_segments):
        cluster_mask = labels == cluster_id
        if cluster_mask.sum() == 0:
            continue

        mask_2d = cluster_mask.reshape(feat_h, feat_w).astype(float)

        # Seed: patch with highest absolute 1st-eigenvec value inside the cluster
        cluster_indices = np.where(cluster_mask)[0]
        best_local = int(np.argmax(np.abs(V[cluster_indices, 0])))
        seed_global = int(cluster_indices[best_local])

        try:
            pred, _, _, _ = detect_box(
                mask_2d, seed_global, dims,
                scales=scales,
                initial_im_size=init_image_size[1:]
            )
        except Exception:
            continue

        preds.append(np.asarray(pred))
        masks.append(mask_2d)
        eigvecs.append(eigvec_full)

    if not preds:
        # Fallback to single ncut so we never return empty-handed
        pred, objects, mask, seed, _, eigvec = ncut(
            feats, dims, scales, init_image_size, tau, eps,
            im_name=im_name, no_binary_graph=no_binary_graph,
            method=method, **method_kwargs
        )
        return [pred], [mask], [eigvec]

    return preds, masks, eigvecs


# ---------------------------------------------------------------------------
# Automatic-k multi-eigenvector NCut — eigengap heuristic (Shi & Malik 2000)
# ---------------------------------------------------------------------------

def _find_k_eigengap(eigenvalues, max_k):
    """
    Eigengap heuristic for automatic cluster-count selection.

    Given the sorted eigenvalues λ₁ ≤ λ₂ ≤ … of the normalised Laplacian,
    the number of clusters k* is the index of the largest gap:

        k* = argmax_{i=1..max_k-1} (λ_{i+1} - λ_i)  + 1

    We skip the trivial first eigenvalue (λ₀ ≈ 0) and search only
    within indices 1..max_k (the non-trivial part of the spectrum).

    Returns an integer k* in [1, max_k].
    """
    # eigenvalues[0] is always ≈ 0 (trivial), start gap search from index 1
    gaps = np.diff(eigenvalues[1:max_k + 1])   # gaps between λ₁..λ_{max_k+1}
    if len(gaps) == 0:
        return 1
    k_star = int(np.argmax(gaps)) + 1   # +1: gap[i] = λ_{i+2}-λ_{i+1}, k = i+1
    return max(1, k_star)


def ncut_auto_k(feats, dims, scales, init_image_size, tau=0, eps=1e-5,
                im_name='', no_binary_graph=False,
                method="baseline", max_k=6,
                kmeans_n_init=10, kmeans_max_iter=300,
                **method_kwargs):
    """
    Automatic-k multi-eigenvector NCut (eigengap heuristic, Shi & Malik 2000).

    The number of clusters is *not* fixed — it is inferred from the
    eigenvalue spectrum of the normalised Laplacian for each image:

      1. Compute the (max_k + 1) smallest eigenvalues/vectors on GPU.
      2. Find the largest gap between consecutive eigenvalues in the
         non-trivial part of the spectrum (indices 1..max_k).
         The cluster count k* is the position of that gap.
      3. Row-normalise the k* spectral embedding vectors.
      4. Run k-means with k = k* to assign patches to clusters.
      5. Convert each cluster to a bounding box.

    Args:
        feats:            Raw ViT features (1, N_tokens, D) — batch + CLS included.
        dims:             [feat_h, feat_w] of the feature map.
        scales:           [scale_h, scale_w] patch→image pixel scaling.
        init_image_size:  Shape of the original (possibly padded) input image.
        tau, eps:         Graph construction thresholds.
        no_binary_graph:  Use soft edge weights instead of binary.
        method:           Affinity method key (see AFFINITY_METHODS).
        max_k:            Upper bound on the number of clusters to consider (default 6).
        kmeans_n_init:    K-means restarts (default 10).
        kmeans_max_iter:  K-means max iterations (default 300).
        **method_kwargs:  Hyperparameters forwarded to the affinity method.

    Returns:
        k_found — int, the automatically selected number of clusters
        preds   — list of np.ndarray bboxes  [x_min, y_min, x_max, y_max]
        masks   — list of 2-D float arrays   (feat_h × feat_w)
        eigvecs — list of 2-D float arrays   (feat_h × feat_w)
    """
    # ---------- Build affinity matrix ----------
    feats_t = feats[0, 1:, :]
    feats_t = feats_t.transpose(0, 1)
    device  = feats_t.device

    feats_t = F.normalize(feats_t, p=2, dim=0)
    A = feats_t.transpose(0, 1) @ feats_t

    if method not in AFFINITY_METHODS:
        raise ValueError(f"Unknown affinity method '{method}'. "
                         f"Choose from: {list(AFFINITY_METHODS)}")
    A = AFFINITY_METHODS[method](A, dims, **method_kwargs)

    if no_binary_graph:
        A = torch.where(A < tau, torch.tensor(eps, device=device, dtype=A.dtype), A)
    else:
        A = (A > tau).to(A.dtype)
        A = torch.where(A == 0, torch.tensor(eps, device=device, dtype=A.dtype), A)

    d_i = A.sum(dim=1)

    # ---------- Normalised Laplacian eigendecomposition on GPU ----------
    D_inv_sqrt = torch.diag(1.0 / d_i.sqrt())
    A_sym = D_inv_sqrt @ A @ D_inv_sqrt
    L_sym = torch.eye(A_sym.shape[0], device=device, dtype=A_sym.dtype) - A_sym
    L_sym = (L_sym + L_sym.T) * 0.5

    eigenvalues, U = torch.linalg.eigh(L_sym)   # ascending order

    # Move to numpy — only need the first max_k+2 values for the gap test
    evals_np = eigenvalues[:max_k + 2].cpu().numpy()

    # ---------- Eigengap: automatically determine k ----------
    k_star = _find_k_eigengap(evals_np, max_k)

    # ---------- Spectral embedding with k* eigenvectors ----------
    V = (D_inv_sqrt @ U[:, 1:k_star + 1]).cpu().numpy()   # (N, k*)

    row_norms = np.linalg.norm(V, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0
    V_norm = V / row_norms

    # ---------- K-means ----------
    if k_star == 1:
        # Trivial: entire graph is one cluster
        labels = np.zeros(V_norm.shape[0], dtype=int)
    else:
        kmeans = KMeans(n_clusters=k_star, n_init=kmeans_n_init,
                        max_iter=kmeans_max_iter, random_state=0)
        labels = kmeans.fit_predict(V_norm)

    # ---------- Map each cluster → bounding box ----------
    feat_h, feat_w = dims
    preds, masks, eigvecs = [], [], []

    eigvec_full = V[:, 0].reshape(feat_h, feat_w)   # 1st dim for visualisation

    for cluster_id in range(k_star):
        cluster_mask = labels == cluster_id
        if cluster_mask.sum() == 0:
            continue

        mask_2d = cluster_mask.reshape(feat_h, feat_w).astype(float)

        cluster_indices = np.where(cluster_mask)[0]
        best_local  = int(np.argmax(np.abs(V[cluster_indices, 0])))
        seed_global = int(cluster_indices[best_local])

        try:
            pred, _, _, _ = detect_box(
                mask_2d, seed_global, dims,
                scales=scales,
                initial_im_size=init_image_size[1:]
            )
        except Exception:
            continue

        preds.append(np.asarray(pred))
        masks.append(mask_2d)
        eigvecs.append(eigvec_full)

    if not preds:
        pred, objects, mask, seed, _, eigvec = ncut(
            feats, dims, scales, init_image_size, tau, eps,
            im_name=im_name, no_binary_graph=no_binary_graph,
            method=method, **method_kwargs
        )
        return 1, [pred], [mask], [eigvec]

    return k_star, preds, masks, eigvecs