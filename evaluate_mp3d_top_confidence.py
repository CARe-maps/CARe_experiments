
import numpy as np
import torch
import os
import open3d
import json
from tqdm import tqdm
import pandas as pd
import clip
import pickle
from einops import einsum
import os
import time

def merge_counter_dicts(dict1, dict2):
    if dict1 == {}: return dict2
    for key in dict2:
        dict1[key] += dict2[key]
    return dict1

def mean_pairwise_kl(probs):
    probs = torch.tensor(probs) # [N, C]
    logprobs = torch.log(probs + 1e-6)
    pairwise_kl = (probs * logprobs).sum (dim = 1) - torch.einsum('ic,jc->ij', probs, logprobs)
    return torch.sum(pairwise_kl) / ((probs.shape[0] - 1) * probs.shape[0])


device = "cuda" if torch.cuda.is_available() else "cpu"
model, transform = clip.load("ViT-L/14", device=device)


def encode_text_feat(texts):
    texts = clip.tokenize(texts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(texts).cpu().numpy()
    return text_features

try: 
    mpcatraw_map, text_features = pickle.load(open("cache/mpcatraw_map.pkl", "rb"))
except: 
    mp3d_categoty_mapping = "/tmp2/htsu/mp3d_pointcloud_gen/openmask3d/matterport_category_mappings.tsv"
    # load category mapping (.tsv)
    category_mapping = pd.read_csv(mp3d_categoty_mapping, sep="    ")
    # raw_categoty to mpcat40index and mpcat40
    category_mapping = category_mapping[["raw_category"]]
    # build an idx to mpcat40 mapping
    mpcatraw_map = list(category_mapping["raw_category"].values)
    # get CLIP features
    text = clip.tokenize(mpcatraw_map).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text).cpu().numpy()
    # normalize text features
    text_features = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)
    os.makedirs("cache", exist_ok=True)
    with open("cache/mpcatraw_map.pkl", "wb") as f:
        pickle.dump([mpcatraw_map, text_features], f)

prediction_root = "./extracted_features/mp3d_0516_25mm"
pc_root = "/tmp2/htsu/mp3d_pointcloud_gen/openmask3d/resources/mp3d_0516_25mm"
label_root = "/tmp2/htsu/mp3d_pointcloud_gen/data/v1/scans"
# scan_id = "8194nk5LbLH"
# region_id = "4"

results = {
    "label_matched_entropy": [],
    "label_mismatched_entropy": [],
    "label_matched_confidence": [],
    "label_mismatched_confidence": [],
    "label_matched_mean_pairwise_kl": [],
    "label_mismatched_mean_pairwise_kl": [],
    "class_dependent_ap": [],
    "class_anymatch_ap": [],
    "class_independent_ap": [],
    "class_dependent_ar": [],
    "class_anymatch_ar": [],
    "class_independent_ar": [],
    "class_dependent_ap_results": {},
    "class_anymatch_ap_results": {},
    "class_independent_ap_results": {},
    "class_dependent_ar_results": {},
    "class_anymatch_ar_results": {},
    "class_independent_ar_results": {}
}
top1hybrid_4_25s = []
top1hybrid_4_50s = []
top1hybrid_4_100s = []
top1hybrid_4_200s = []
top1hybrid_4_400s = []
top1hybrid_4_800s = []
top1hybrid_8_25s = []
top1hybrid_8_50s = []
top1hybrid_8_100s = []
top1hybrid_8_200s = []
top1hybrid_8_400s = []
top1hybrid_8_800s = []
top1hybrid_16_25s = []
top1hybrid_16_50s = []
top1hybrid_16_100s = []
top1hybrid_16_200s = []
top1hybrid_16_400s = []
top1hybrid_16_800s = []
top1hybrid_40_25s = []
top1hybrid_40_50s = []
top1hybrid_40_100s = []
top1hybrid_40_200s = []
top1hybrid_40_400s = []
top1hybrid_40_800s = []
top1s = []
top2s = []
top1s_random = []
top1s_random_from_top_2 = []
top1s_random_from_top_4 = []
top1s_random_from_top_8 = []
top1s_random_from_top_16 = []
top1s_random_from_top_40 = []
top1ent1s = []
top1ent2s = []
top1ent4s = []
top1ent8s = []
top1ent16s = []
top1ent40s =    []
top1ent1s_min = []
top1ent2s_min = []
top1ent4s_min = []
top1ent8s_min = []
top1ent16s_min = []
top1ent40s_min = []
top1std1s = []
top1std2s = []
top1std4s = []
top1std8s = []
top1std16s = []
top1std40s = []
top1stderr1s = []
top1stderr2s = []
top1stderr4s = []
top1stderr8s = []
top1stderr16s = []
top1stderr40s = []
top1std1s_min = []
top1std2s_min = []
top1std4s_min = []
top1std8s_min = []
top1std16s_min = []
top1std40s_min = []
top1stderr1s_min = []
top1stderr2s_min = []
top1stderr4s_min = []
top1stderr8s_min = []
top1stderr16s_min = []
top1stderr40s_min = []
top1andkl1s = []
top1andkl2s = []
top1andkl4s = []
top1andkl8s = []
top1andkl16s = []
top1andkl40s = []
top1andkl1s_min = []
top1andkl2s_min = []
top1andkl4s_min = []
top1andkl8s_min = []
top1andkl16s_min = []
top1andkl40s_min = []
justs_random = []
justs_random_from_top1_confidence = []
justs_random_from_top2_confidence = []
justs_random_from_top4_confidence = []
justs_random_from_top8_confidence = []
justs_random_from_top16_confidence = []
justs_random_from_top1_entropy = []
justs_random_from_top2_entropy = []
justs_random_from_top4_entropy = []
justs_random_from_top8_entropy = []
justs_random_from_top16_entropy = []
justs_random_from_top1_entropy_min = []
justs_random_from_top2_entropy_min = []
justs_random_from_top4_entropy_min = []
justs_random_from_top8_entropy_min = []
justs_random_from_top16_entropy_min = []
justs_random_from_top1_stderr = []
justs_random_from_top2_stderr = []
justs_random_from_top4_stderr = []
justs_random_from_top8_stderr = []
justs_random_from_top16_stderr = []
justs_random_from_top1_stderr_min = []
justs_random_from_top2_stderr_min = []
justs_random_from_top4_stderr_min = []
justs_random_from_top8_stderr_min = []
justs_random_from_top16_stderr_min = []
justs_random_from_top1_kl = []
justs_random_from_top2_kl = []
justs_random_from_top4_kl = []
justs_random_from_top8_kl = []
justs_random_from_top16_kl = []
justs_random_from_top1_kl_min = []
justs_random_from_top2_kl_min = []
justs_random_from_top4_kl_min = []
justs_random_from_top8_kl_min = []
justs_random_from_top16_kl_min = []

time_recordings = {}

evaluated_instances = 0
evaluated_rooms = 0
scan_ids = os.listdir(prediction_root)
for i_s, scan_id in enumerate(scan_ids):
    # if i_s != 0: break
    region_ids = os.listdir(os.path.join(prediction_root, scan_id))
    print(region_ids)
    
    for region_id in tqdm(sorted(region_ids)):
        print(f"Processing scan {i_s+1}/{len(scan_ids)}: {scan_id} {region_id}")
        try: 
            # Load prediction
            pred_masks = torch.load(os.path.join(prediction_root, scan_id, region_id, "pointcloud_masks.pt"))
            pred_mask2feats = np.load(os.path.join(prediction_root, scan_id, region_id, "pointcloud_openmask3d_features.npy"))
            pred_pc = open3d.io.read_point_cloud(os.path.join(pc_root, scan_id, region_id, "pointcloud.ply"))
        except:
            # somethiong wrong with the prediction, skip with a warning
            print(f"[Warning] Error loading prediction for {scan_id} {region_id}")
            continue
        # build gt labels
        gt_pc = open3d.io.read_point_cloud(os.path.join(label_root, scan_id, "region_segmentations", f"region{region_id}.ply"))
        gt_vertices = json.load(open(os.path.join(label_root, scan_id, "region_segmentations", f"region{region_id}.vsegs.json")))["segIndices"]
        gt_instances = json.load(open(os.path.join(label_root, scan_id, "region_segmentations", f"region{region_id}.semseg.json")))["segGroups"]

        
        # normalize pred features
        # pred_mask2feats_mean = np.mean(pred_mask2feats, axis=1)
        # pred_mask2feats_mean = pred_mask2feats_mean / np.linalg.norm(pred_mask2feats_mean, axis=1, keepdims=True)
        # pred_logits_mean = einsum(text_features, pred_mask2feats_mean, "i c, j c ->j i")
        # pred_probs_mean = torch.softmax(torch.tensor(pred_logits_mean), dim=-1).numpy()
        pred_mask2feats = pred_mask2feats / np.linalg.norm(pred_mask2feats, axis=2, keepdims=True)
        temperature = 5
        pred_logits = einsum(text_features, pred_mask2feats, "i c, j v c ->j v i") * temperature
        pred_probs = torch.softmax(torch.tensor(pred_logits), dim=-1).numpy()
        max_probs = np.max(pred_probs, axis=-1)
        pred_nonzero_views = np.isnan(max_probs) == False
        # print(pred_nonzero_views[:10])

        # create pred_instances
        pred_instances = []
        pred_features = []
        skip_cnt = 0
        for i, mask in enumerate(pred_masks.T):
            # throw away all-zero masks
            if np.sum(pred_nonzero_views[i]) == 0:
                skip_cnt += 1
                continue
            pred_features.append(pred_mask2feats[i])
            non_zero_views = np.where(pred_nonzero_views[i])[0]
            non_zero_features = pred_mask2feats[i][non_zero_views]
            pred_variace = np.var(non_zero_features, axis=0) 
            total_std = np.sqrt(np.sum(pred_variace)) if len(non_zero_views) > 1 else float("inf")
            total_stderr = total_std / np.sqrt(len(non_zero_views))
            mask = np.where(mask == 1)[0]
            points = np.array(pred_pc.points)[mask]
            logit = pred_logits[i][pred_nonzero_views[i]]
            prob = pred_probs[i][pred_nonzero_views[i]]
            entropy = -np.sum(prob * np.log(prob + 1e-6), axis=-1)
            cls_idx = np.argmax(logit, axis=-1)
            logit_mean = np.mean(logit, axis=0)
            prob_mean = torch.softmax(torch.tensor(logit_mean), dim=-1).numpy()
            instance = {
                "points": points,
                "cm": np.mean(points, axis=0),
                "indices": mask,
                "logit": logit,
                "prob": prob,
                "total_std": total_std,
                "total_stderr": total_stderr,
                "cls_idx": cls_idx,
                "cls_text": [mpcatraw_map[idx] for idx in cls_idx],
                "cls_entropy": entropy,
                "cls_confidence" : np.max(prob, axis=-1),
                "mean_pairwise_kl": mean_pairwise_kl(prob),
                "mean_logit": logit_mean,
                "mean_prob": prob_mean,
                "mean_cls_idx": np.argmax(logit_mean),
                "mean_cls_text": mpcatraw_map[np.argmax(logit_mean)],
                "mean_cls_entropy": -np.sum(prob_mean * np.log(prob_mean + 1e-6), axis=-1),
            }
            pred_instances.append(instance)
        pred_features = np.array(pred_features)
        pred_probs_mean = np.array([instance["mean_prob"] for instance in pred_instances])



        print(f"skiped {skip_cnt} pred instances")
        print(f"Number of processed pred instances: {len(pred_instances)}")

        vertex2point = {}
        vertex2idx = {}
        assert gt_pc.points == pred_pc.points

        for point_idx in range(len(gt_vertices)):
            vertex_idx = gt_vertices[point_idx]
            try:
                vertex2point[vertex_idx].append(gt_pc.points[point_idx])
                vertex2idx[vertex_idx].append(point_idx)
            except:
                vertex2point[vertex_idx] = [gt_pc.points[point_idx]]
                vertex2idx[vertex_idx] = [point_idx]

        for i, instance in enumerate(gt_instances):
            vertices = instance["segments"]
            points = []
            indices = []    
            for vertex in vertices:
                try:
                    points.extend(vertex2point[vertex])
                    indices.extend(vertex2idx[vertex])
                except:
                    # if vertex not in vertex2point:
                    pass    
            instance["points"] = np.array(points)
            instance["cm"] = np.mean(points, axis=0)    
            instance["indices"] = np.array(indices)
            # update label
            instance["raw_label"] = instance["label"]
            instance["raw_feat"] = encode_text_feat([instance["label"]])
            instance["raw_label_idx"] = mpcatraw_map.index(instance["label"])

        # print(instance.keys())
        print("Number of GT instances: ", len(gt_instances))
        evaluated_instances += len(gt_instances)

        def mean_top_k_acc(k, iou_threshold=0.25):
            results = []
            for gt_instance in gt_instances:
                # raw_feat = gt_instance["raw_feat"]
                # scores = (pred_features @ raw_feat.T)[:, :, 0]
                gt_ind = gt_instance["raw_label_idx"]
                mean_scores = pred_probs_mean[:, gt_ind]
                # mean_scores = np.mean(scores, axis=1)
                sorted_idx = np.argsort(mean_scores)[::-1]
                top_k = sorted_idx[:k]
                top_k_instances = [pred_instances[i] for i in top_k]
                success = False
                for i, pred_instance in enumerate(top_k_instances):
                    gt_indices = gt_instance["indices"]
                    pred_indices = pred_instance["indices"]
                    intersection = np.intersect1d(gt_indices, pred_indices)
                    union = np.union1d(gt_indices, pred_indices)
                    iou = len(intersection) / len(union)
                    if iou > iou_threshold:
                        success = True
                        break
                results.append(success)
            return np.mean(results)

        def mean_top_k_acc_random(k, iou_threshold=0.25):
            results = []
            for gt_instance in gt_instances:
                top_k = np.random.choice(len(pred_instances), k, replace=False)
                top_k_instances = [pred_instances[i] for i in top_k]
                success = False
                for i, pred_instance in enumerate(top_k_instances):
                    gt_indices = gt_instance["indices"]
                    pred_indices = pred_instance["indices"]
                    intersection = np.intersect1d(gt_indices, pred_indices)
                    union = np.union1d(gt_indices, pred_indices)
                    iou = len(intersection) / len(union)
                    if iou > iou_threshold:
                        success = True
                        break
                results.append(success)
            return np.mean(results)
        
        def mean_top_k_acc_top_n_entropy(k, n, iou_threshold=0.25):
            results = []
            for gt_instance in gt_instances:
                # raw_feat = gt_instance["raw_feat"]
                # scores = (pred_features @ raw_feat.T)[:, :, 0]
                gt_ind = gt_instance["raw_label_idx"]
                mean_scores = pred_probs_mean[:, gt_ind]
                # mean_scores = np.mean(scores, axis=1)
                sorted_idx = np.argsort(mean_scores)[::-1]
                top_n = sorted_idx[:n]
                entropies = [pred_instances[i]["mean_cls_entropy"] for i in top_n]
                # sort by entropy 
                sorted_idx = np.argsort(entropies)[::-1]
                top_k = [top_n[i] for i in sorted_idx[:k]]
                top_k_instances = [pred_instances[i] for i in top_k]
                success = False
                for i, pred_instance in enumerate(top_k_instances):
                    gt_indices = gt_instance["indices"]
                    pred_indices = pred_instance["indices"]
                    intersection = np.intersect1d(gt_indices, pred_indices)
                    union = np.union1d(gt_indices, pred_indices)
                    iou = len(intersection) / len(union)
                    if iou > iou_threshold:
                        success = True
                        break
                results.append(success)
            return np.mean(results)

        def mean_top_1_acc_top_n_entropy(n, iou_threshold=0.25, min_ent=False, skip_top1=False):
            results = []
            for gt_instance in gt_instances:
                # raw_feat = gt_instance["raw_feat"]
                # scores = (pred_features @ raw_feat.T)[:, :, 0]
                gt_ind = gt_instance["raw_label_idx"]
                mean_scores = pred_probs_mean[:, gt_ind]
                # mean_scores = np.mean(scores, axis=1)
                sorted_idx = np.argsort(mean_scores)[::-1]
                if not skip_top1:
                    top_1 = sorted_idx[0]
                    bottom = sorted_idx[1:]
                    topn = bottom[:n]
                    entropies = [pred_instances[i]["mean_cls_entropy"] for i in topn]
                    # sort by entropy 
                    sorted_idx = np.argsort(entropies)[::-1]
                    top_1_ent = topn[sorted_idx[0]] if not min_ent else topn[sorted_idx[-1]]
                    top_k_instances = [pred_instances[top_1]] + [pred_instances[top_1_ent]]
                else:
                    topn = sorted_idx[:n]
                    entropies = [pred_instances[i]["mean_cls_entropy"] for i in topn]
                    # sort by entropy
                    sorted_idx = np.argsort(entropies)[::-1]
                    top_1_ent = topn[sorted_idx[0]] if not min_ent else topn[sorted_idx[-1]]
                    top_k_instances = [pred_instances[top_1_ent]]

                success = False
                for i, pred_instance in enumerate(top_k_instances):
                    gt_indices = gt_instance["indices"]
                    pred_indices = pred_instance["indices"]
                    intersection = np.intersect1d(gt_indices, pred_indices)
                    union = np.union1d(gt_indices, pred_indices)
                    iou = len(intersection) / len(union)
                    if iou > iou_threshold:
                        success = True
                        break
                results.append(success)
            return np.mean(results)

        def mean_top_1_and_random(iou_threshold=0.25, skip_top1=False):
            results = []
            for gt_instance in gt_instances:
                # raw_feat = gt_instance["raw_feat"]
                # scores = (pred_features @ raw_feat.T)[:, :, 0]
                gt_ind = gt_instance["raw_label_idx"]
                mean_scores = pred_probs_mean[:, gt_ind]
                # mean_scores = np.mean(scores, axis=1)
                sorted_idx = np.argsort(mean_scores)[::-1]
                if not skip_top1:
                    top_1 = sorted_idx[0]
                    top_k_instances = [pred_instances[top_1]] + [pred_instances[i] for i in sorted_idx[1:]]
                else:
                    top_k_instances = [pred_instances[i] for i in sorted_idx]

                score = 0
                for i, pred_instance in enumerate(top_k_instances):
                    gt_indices = gt_instance["indices"]
                    pred_indices = pred_instance["indices"]
                    intersection = np.intersect1d(gt_indices, pred_indices)
                    union = np.union1d(gt_indices, pred_indices)
                    iou = len(intersection) / len(union)
                    if iou > iou_threshold:
                        if not skip_top1 and i == 0:
                            score += 1
                            break
                        else:
                            score += 1 / (len(top_k_instances) - 1) # expected value
                results.append(score)
            return np.mean(results)

        def mean_top_1_and_random_from_top_n(n, iou_threshold=0.25, skip_top1=False):
            results = []
            for gt_instance in gt_instances:
                # raw_feat = gt_instance["raw_feat"]
                # scores = (pred_features @ raw_feat.T)[:, :, 0]
                gt_ind = gt_instance["raw_label_idx"]
                mean_scores = pred_probs_mean[:, gt_ind]
                # mean_scores = np.mean(scores, axis=1)
                sorted_idx = np.argsort(mean_scores)[::-1]
                if not skip_top1:
                    sorted_idx = sorted_idx[:n+1]
                    top_1 = sorted_idx[0]
                    top_k_instances = [pred_instances[top_1]] + [pred_instances[i] for i in sorted_idx[1:]]
                else:
                    top_k_instances = [pred_instances[i] for i in sorted_idx[:n]]
                # assert len(top_k_instances) == n+1
                score = 0
                for i, pred_instance in enumerate(top_k_instances):
                    gt_indices = gt_instance["indices"]
                    pred_indices = pred_instance["indices"]
                    intersection = np.intersect1d(gt_indices, pred_indices)
                    union = np.union1d(gt_indices, pred_indices)
                    iou = len(intersection) / len(union)
                    if iou > iou_threshold:
                        if i == 0 and not skip_top1:
                            score += 1
                            break
                        else:   
                            score += 1 / n
                results.append(score)
            return np.mean(results)
        
        def mean_top_1_acc_top_n_std(n, iou_threshold=0.25, min_ent=False):
            results = []
            for gt_instance in gt_instances:
                # raw_feat = gt_instance["raw_feat"]
                # scores = (pred_features @ raw_feat.T)[:, :, 0]
                gt_ind = gt_instance["raw_label_idx"]
                mean_scores = pred_probs_mean[:, gt_ind]
                # mean_scores = np.mean(scores, axis=1)
                sorted_idx = np.argsort(mean_scores)[::-1]
                top_1 = sorted_idx[0]
                bottom = sorted_idx[1:]
                topn = bottom[:n]
                stds = [pred_instances[i]["total_std"] for i in topn]
                # sort by entropy 
                sorted_idx = np.argsort(stds)[::-1]
                top_1_std = topn[sorted_idx[0]] if not min_ent else topn[sorted_idx[-1]]
                top_k_instances = [pred_instances[top_1]] + [pred_instances[top_1_std]]
                success = False
                for i, pred_instance in enumerate(top_k_instances):
                    gt_indices = gt_instance["indices"]
                    pred_indices = pred_instance["indices"]
                    intersection = np.intersect1d(gt_indices, pred_indices)
                    union = np.union1d(gt_indices, pred_indices)
                    iou = len(intersection) / len(union)
                    if iou > iou_threshold:
                        success = True
                        break
                results.append(success)
            return np.mean(results)
        
        def mean_top_1_acc_top_n_stderr(n, iou_threshold=0.25, min_ent=False, skip_top1=False):
            results = []
            for gt_instance in gt_instances:
                # raw_feat = gt_instance["raw_feat"]
                # scores = (pred_features @ raw_feat.T)[:, :, 0]
                gt_ind = gt_instance["raw_label_idx"]
                mean_scores = pred_probs_mean[:, gt_ind]
                # mean_scores = np.mean(scores, axis=1)
                sorted_idx = np.argsort(mean_scores)[::-1]
                if not skip_top1:
                    top_1 = sorted_idx[0]
                    bottom = sorted_idx[1:]
                    topn = bottom[:n]
                    stds = [pred_instances[i]["total_stderr"] for i in topn]
                    # sort by entropy 
                    sorted_idx = np.argsort(stds)[::-1]
                    top_1_std = topn[sorted_idx[0]] if not min_ent else topn[sorted_idx[-1]]
                    top_k_instances = [pred_instances[top_1]] + [pred_instances[top_1_std]]
                else:
                    topn = sorted_idx[:n]
                    stds = [pred_instances[i]["total_stderr"] for i in topn]
                    # sort by entropy
                    sorted_idx = np.argsort(stds)[::-1]
                    top_1_std = topn[sorted_idx[0]] if not min_ent else topn[sorted_idx[-1]]
                    top_k_instances = [pred_instances[top_1_std]]
                success = False
                for i, pred_instance in enumerate(top_k_instances):
                    gt_indices = gt_instance["indices"]
                    pred_indices = pred_instance["indices"]
                    intersection = np.intersect1d(gt_indices, pred_indices)
                    union = np.union1d(gt_indices, pred_indices)
                    iou = len(intersection) / len(union)
                    if iou > iou_threshold:
                        success = True
                        break
                results.append(success)
            return np.mean(results)

        def mean_top_1_acc_top_n_kl(n, iou_threshold=0.25, min_ent=False, skip_top1=False):
            results = []
            for gt_instance in gt_instances:
                # raw_feat = gt_instance["raw_feat"]
                # scores = (pred_features @ raw_feat.T)[:, :, 0]
                gt_ind = gt_instance["raw_label_idx"]
                mean_scores = pred_probs_mean[:, gt_ind]
                # mean_scores = np.mean(scores, axis=1)
                sorted_idx = np.argsort(mean_scores)[::-1]
                if not skip_top1:
                    top_1 = sorted_idx[0]
                    bottom = sorted_idx[1:]
                    topn = bottom[:n]
                    kls = [mean_pairwise_kl(pred_probs[i]) for i in topn]
                    # sort by entropy 
                    sorted_idx = np.argsort(kls)[::-1]
                    top_1_kl = topn[sorted_idx[0]] if not min_ent else topn[sorted_idx[-1]]
                    top_k_instances = [pred_instances[top_1]] + [pred_instances[top_1_kl]]
                else:
                    topn = sorted_idx[:n]
                    kls = [mean_pairwise_kl(pred_probs[i]) for i in topn]
                    # sort by entropy
                    sorted_idx = np.argsort(kls)[::-1]
                    top_1_kl = topn[sorted_idx[0]] if not min_ent else topn[sorted_idx[-1]]
                    top_k_instances = [pred_instances[top_1_kl]]
                success = False
                for i, pred_instance in enumerate(top_k_instances):
                    gt_indices = gt_instance["indices"]
                    pred_indices = pred_instance["indices"]
                    intersection = np.intersect1d(gt_indices, pred_indices)
                    union = np.union1d(gt_indices, pred_indices)
                    iou = len(intersection) / len(union)
                    if iou > iou_threshold:
                        success = True
                        break
                results.append(success)
            return np.mean(results)

        def mean_top_1_acc_top_n_hybrid(n, iou_threshold=0.25, kl_ratio=1.0):
            results = []
            for gt_instance in gt_instances:
                # raw_feat = gt_instance["raw_feat"]
                # scores = (pred_features @ raw_feat.T)[:, :, 0]
                gt_ind = gt_instance["raw_label_idx"]
                mean_scores = pred_probs_mean[:, gt_ind]
                # mean_scores = np.mean(scores, axis=1)
                sorted_idx = np.argsort(mean_scores)[::-1]
                top_1 = sorted_idx[0]
                bottom = sorted_idx[1:]
                topn = bottom[:n]
                entropies = [pred_instances[i]["mean_cls_entropy"] for i in topn]
                kls = [mean_pairwise_kl(pred_probs[i]) for i in topn]
                rank_ent = np.argsort(entropies)[::-1].argsort() # the larger the better
                rank_kl = np.argsort(kls).argsort() # the smaller the better
                combined_rank = rank_ent + kl_ratio * rank_kl
                # combine the ranks of entropy and kl
                top_1_hybrid = topn[np.argmin(combined_rank)]
                top_k_instances = [pred_instances[top_1]] + [pred_instances[top_1_hybrid]]
                success = False
                for i, pred_instance in enumerate(top_k_instances):
                    gt_indices = gt_instance["indices"]
                    pred_indices = pred_instance["indices"]
                    intersection = np.intersect1d(gt_indices, pred_indices)
                    union = np.union1d(gt_indices, pred_indices)
                    iou = len(intersection) / len(union)
                    if iou > iou_threshold:
                        success = True
                        break
                results.append(success)
            return np.mean(results) 

        # add recordings
        t0 = time.time()    
        top1 = mean_top_k_acc(1)
        time_recordings["top1s"] = time_recordings.get("top1s", []) + [time.time() - t0]
        t0 = time.time()
        top1_random = mean_top_1_and_random()
        time_recordings["top1s_random"] = time_recordings.get("top1s_random", []) + [time.time() - t0]
        t0 = time.time()
        top1_random_from_top_2 = mean_top_1_and_random_from_top_n(2)
        time_recordings["top1s_random_from_top_2"] = time_recordings.get("top1s_random_from_top_2", []) + [time.time() - t0]
        t0 = time.time()
        top1_random_from_top_4 = mean_top_1_and_random_from_top_n(4)
        time_recordings["top1s_random_from_top_4"] = time_recordings.get("top1s_random_from_top_4", []) + [time.time() - t0]
        t0 = time.time()
        top1_random_from_top_8 = mean_top_1_and_random_from_top_n(8)
        time_recordings["top1s_random_from_top_8"] = time_recordings.get("top1s_random_from_top_8", []) + [time.time() - t0]
        t0 = time.time()
        top1_random_from_top_16 = mean_top_1_and_random_from_top_n(16)
        time_recordings["top1s_random_from_top_16"] = time_recordings.get("top1s_random_from_top_16", []) + [time.time() - t0]
        t0 = time.time()
        top1_random_from_top_40 = mean_top_1_and_random_from_top_n(40)
        time_recordings["top1s_random_from_top_40"] = time_recordings.get("top1s_random_from_top_40", []) + [time.time() - t0]
        t0 = time.time()
        top2 = mean_top_k_acc(2)
        time_recordings["top2s"] = time_recordings.get("top2s", []) + [time.time() - t0]
        
        t0 = time.time()
        top1andentropy_1 = mean_top_1_acc_top_n_entropy(1)
        time_recordings["top1andentropy_1s"] = time_recordings.get("top1andentropy_1s", []) + [time.time() - t0]
        t0 = time.time()
        top1andentropy_2 = mean_top_1_acc_top_n_entropy(2)
        time_recordings["top1andentropy_2s"] = time_recordings.get("top1andentropy_2s", []) + [time.time() - t0]
        t0 = time.time()
        top1andentropy_4 = mean_top_1_acc_top_n_entropy(4)
        time_recordings["top1andentropy_4s"] = time_recordings.get("top1andentropy_4s", []) + [time.time() - t0]
        t0 = time.time()
        top1andentropy_8 = mean_top_1_acc_top_n_entropy(8)
        time_recordings["top1andentropy_8s"] = time_recordings.get("top1andentropy_8s", []) + [time.time() - t0]
        t0 = time.time()   
        top1andentropy_16= mean_top_1_acc_top_n_entropy(16)
        time_recordings["top1andentropy_16s"] = time_recordings.get("top1andentropy_16s", []) + [time.time() - t0]
        t0 = time.time()
        top1andentropy_40= mean_top_1_acc_top_n_entropy(40)
        time_recordings["top1andentropy_40s"] = time_recordings.get("top1andentropy_40s", []) + [time.time() - t0]

        t0 = time.time()
        top1andentropy_1_min = mean_top_1_acc_top_n_entropy(1, min_ent=True)
        time_recordings["top1andentropy_1s_min"] = time_recordings.get("top1andentropy_1s_min", []) + [time.time() - t0]
        t0 = time.time()
        top1andentropy_2_min = mean_top_1_acc_top_n_entropy(2, min_ent=True)
        time_recordings["top1andentropy_2s_min"] = time_recordings.get("top1andentropy_2s_min", []) + [time.time() - t0]
        t0 = time.time()
        top1andentropy_4_min = mean_top_1_acc_top_n_entropy(4, min_ent=True)
        time_recordings["top1andentropy_4s_min"] = time_recordings.get("top1andentropy_4s_min", []) + [time.time() - t0]
        t0 = time.time()
        top1andentropy_8_min = mean_top_1_acc_top_n_entropy(8, min_ent=True)
        time_recordings["top1andentropy_8s_min"] = time_recordings.get("top1andentropy_8s_min", []) + [time.time() - t0]
        t0 = time.time()
        top1andentropy_16_min = mean_top_1_acc_top_n_entropy(16, min_ent=True)
        time_recordings["top1andentropy_16s_min"] = time_recordings.get("top1andentropy_16s_min", []) + [time.time() - t0]
        t0 = time.time()
        top1andentropy_40_min = mean_top_1_acc_top_n_entropy(40, min_ent=True)
        time_recordings["top1andentropy_40s_min"] = time_recordings.get("top1andentropy_40s_min", []) + [time.time() - t0]

        t0 = time.time()
        top1andstd_1 = mean_top_1_acc_top_n_std(1)
        time_recordings["top1andstd_1s"] = time_recordings.get("top1andstd_1s", []) + [time.time() - t0]
        t0 = time.time()
        top1andstd_2 = mean_top_1_acc_top_n_std(2)
        time_recordings["top1andstd_2s"] = time_recordings.get("top1andstd_2s", []) + [time.time() - t0]
        t0 = time.time()
        top1andstd_4 = mean_top_1_acc_top_n_std(4)
        time_recordings["top1andstd_4s"] = time_recordings.get("top1andstd_4s", []) + [time.time() - t0]
        t0 = time.time()
        top1andstd_8 = mean_top_1_acc_top_n_std(8)
        time_recordings["top1andstd_8s"] = time_recordings.get("top1andstd_8s", []) + [time.time() - t0]
        t0 = time.time()
        top1andstd_16 = mean_top_1_acc_top_n_std(16)
        time_recordings["top1andstd_16s"] = time_recordings.get("top1andstd_16s", []) + [time.time() - t0]
        t0 = time.time()
        top1andstd_40 = mean_top_1_acc_top_n_std(40)
        time_recordings["top1andstd_40s"] = time_recordings.get("top1andstd_40s", []) + [time.time() - t0]

        t0 = time.time()
        top1andstd_1_min = mean_top_1_acc_top_n_std(1, min_ent=True)
        time_recordings["top1andstd_1s_min"] = time_recordings.get("top1andstd_1s_min", []) + [time.time() - t0]
        t0 = time.time()
        top1andstd_2_min = mean_top_1_acc_top_n_std(2, min_ent=True)
        time_recordings["top1andstd_2s_min"] = time_recordings.get("top1andstd_2s_min", []) + [time.time() - t0]
        t0 = time.time()
        top1andstd_4_min = mean_top_1_acc_top_n_std(4, min_ent=True)
        time_recordings["top1andstd_4s_min"] = time_recordings.get("top1andstd_4s_min", []) + [time.time() - t0]
        t0 = time.time()
        top1andstd_8_min = mean_top_1_acc_top_n_std(8, min_ent=True)
        time_recordings["top1andstd_8s_min"] = time_recordings.get("top1andstd_8s_min", []) + [time.time() - t0]
        t0 = time.time()
        top1andstd_16_min = mean_top_1_acc_top_n_std(16, min_ent=True)
        time_recordings["top1andstd_16s_min"] = time_recordings.get("top1andstd_16s_min", []) + [time.time() - t0]
        t0 = time.time()
        top1andstd_40_min = mean_top_1_acc_top_n_std(40, min_ent=True)
        time_recordings["top1andstd_40s_min"] = time_recordings.get("top1andstd_40s_min", []) + [time.time() - t0]

        t0 = time.time()
        top1andstderr_1 = mean_top_1_acc_top_n_stderr(1)
        time_recordings["top1andstderr_1s"] = time_recordings.get("top1andstderr_1s", []) + [time.time() - t0]
        t0 = time.time()
        top1andstderr_2 = mean_top_1_acc_top_n_stderr(2)
        time_recordings["top1andstderr_2s"] = time_recordings.get("top1andstderr_2s", []) + [time.time() - t0]
        t0 = time.time()
        top1andstderr_4 = mean_top_1_acc_top_n_stderr(4)
        time_recordings["top1andstderr_4s"] = time_recordings.get("top1andstderr_4s", []) + [time.time() - t0]
        t0 = time.time()
        top1andstderr_8 = mean_top_1_acc_top_n_stderr(8)
        time_recordings["top1andstderr_8s"] = time_recordings.get("top1andstderr_8s", []) + [time.time() - t0]
        t0 = time.time()
        top1andstderr_16 = mean_top_1_acc_top_n_stderr(16)
        time_recordings["top1andstderr_16s"] = time_recordings.get("top1andstderr_16s", []) + [time.time() - t0]
        t0 = time.time()
        top1andstderr_40 = mean_top_1_acc_top_n_stderr(40)
        time_recordings["top1andstderr_40s"] = time_recordings.get("top1andstderr_40s", []) + [time.time() - t0]
        
        t0 = time.time()
        top1andstderr_1_min = mean_top_1_acc_top_n_stderr(1, min_ent=True)
        time_recordings["top1andstderr_1s_min"] = time_recordings.get("top1andstderr_1s_min", []) + [time.time() - t0]
        t0 = time.time()
        top1andstderr_2_min = mean_top_1_acc_top_n_stderr(2, min_ent=True)
        time_recordings["top1andstderr_2s_min"] = time_recordings.get("top1andstderr_2s_min", []) + [time.time() - t0]
        t0 = time.time()
        top1andstderr_4_min = mean_top_1_acc_top_n_stderr(4, min_ent=True)
        time_recordings["top1andstderr_4s_min"] = time_recordings.get("top1andstderr_4s_min", []) + [time.time() - t0]
        t0 = time.time()
        top1andstderr_8_min = mean_top_1_acc_top_n_stderr(8, min_ent=True)
        time_recordings["top1andstderr_8s_min"] = time_recordings.get("top1andstderr_8s_min", []) + [time.time() - t0]
        t0 = time.time()
        top1andstderr_16_min = mean_top_1_acc_top_n_stderr(16, min_ent=True)
        time_recordings["top1andstderr_16s_min"] = time_recordings.get("top1andstderr_16s_min", []) + [time.time() - t0]
        t0 = time.time()
        top1andstderr_40_min = mean_top_1_acc_top_n_stderr(40, min_ent=True)
        time_recordings["top1andstderr_40s_min"] = time_recordings.get("top1andstderr_40s_min", []) + [time.time() - t0]

        t0 = time.time()
        top1andkl_1 = mean_top_1_acc_top_n_kl(1)
        time_recordings["top1andkl_1s"] = time_recordings.get("top1andkl_1s", []) + [time.time() - t0]
        t0 = time.time()
        top1andkl_2 = mean_top_1_acc_top_n_kl(2)
        time_recordings["top1andkl_2s"] = time_recordings.get("top1andkl_2s", []) + [time.time() - t0]
        t0 = time.time()
        top1andkl_4 = mean_top_1_acc_top_n_kl(4)
        time_recordings["top1andkl_4s"] = time_recordings.get("top1andkl_4s", []) + [time.time() - t0]
        t0 = time.time()
        top1andkl_8 = mean_top_1_acc_top_n_kl(8)
        time_recordings["top1andkl_8s"] = time_recordings.get("top1andkl_8s", []) + [time.time() - t0]
        t0 = time.time()
        top1andkl_16 = mean_top_1_acc_top_n_kl(16)
        time_recordings["top1andkl_16s"] = time_recordings.get("top1andkl_16s", []) + [time.time() - t0]
        t0 = time.time()
        top1andkl_40 = mean_top_1_acc_top_n_kl(40)
        time_recordings["top1andkl_40s"] = time_recordings.get("top1andkl_40s", []) + [time.time() - t0]    

        t0 = time.time()
        top1andkl_1_min = mean_top_1_acc_top_n_kl(1, min_ent=True)
        time_recordings["top1andkl_1s_min"] = time_recordings.get("top1andkl_1s_min", []) + [time.time() - t0]
        t0 = time.time()
        top1andkl_2_min = mean_top_1_acc_top_n_kl(2, min_ent=True)
        time_recordings["top1andkl_2s_min"] = time_recordings.get("top1andkl_2s_min", []) + [time.time() - t0]
        t0 = time.time()
        top1andkl_4_min = mean_top_1_acc_top_n_kl(4, min_ent=True)
        time_recordings["top1andkl_4s_min"] = time_recordings.get("top1andkl_4s_min", []) + [time.time() - t0]
        t0 = time.time()
        top1andkl_8_min = mean_top_1_acc_top_n_kl(8, min_ent=True)
        time_recordings["top1andkl_8s_min"] = time_recordings.get("top1andkl_8s_min", []) + [time.time() - t0]
        t0 = time.time()
        top1andkl_16_min = mean_top_1_acc_top_n_kl(16, min_ent=True)
        time_recordings["top1andkl_16s_min"] = time_recordings.get("top1andkl_16s_min", []) + [time.time() - t0]
        t0 = time.time()
        top1andkl_40_min = mean_top_1_acc_top_n_kl(40, min_ent=True)
        time_recordings["top1andkl_40s_min"] = time_recordings.get("top1andkl_40s_min", []) + [time.time() - t0]

        time_recordings["total_gt_entries"] = time_recordings.get("total_gt_entries", []) + [len(gt_instances)]

        # skip top1 for below
        just_random = mean_top_1_and_random(skip_top1=True)

        just_random_from_top1_confidence = mean_top_1_and_random_from_top_n(1, skip_top1=True)
        just_random_from_top2_confidence = mean_top_1_and_random_from_top_n(2, skip_top1=True)
        just_random_from_top4_confidence = mean_top_1_and_random_from_top_n(4, skip_top1=True)
        just_random_from_top8_confidence = mean_top_1_and_random_from_top_n(8, skip_top1=True)
        just_random_from_top16_confidence = mean_top_1_and_random_from_top_n(16, skip_top1=True)

        just_random_from_top1_entropy = mean_top_1_acc_top_n_entropy(1, skip_top1=True)
        just_random_from_top2_entropy = mean_top_1_acc_top_n_entropy(2, skip_top1=True)
        just_random_from_top4_entropy = mean_top_1_acc_top_n_entropy(4, skip_top1=True)
        just_random_from_top8_entropy = mean_top_1_acc_top_n_entropy(8, skip_top1=True)
        just_random_from_top16_entropy = mean_top_1_acc_top_n_entropy(16, skip_top1=True)

        just_random_from_top1_entropy_min = mean_top_1_acc_top_n_entropy(1, min_ent=True, skip_top1=True)
        just_random_from_top2_entropy_min = mean_top_1_acc_top_n_entropy(2, min_ent=True, skip_top1=True)
        just_random_from_top4_entropy_min = mean_top_1_acc_top_n_entropy(4, min_ent=True, skip_top1=True)
        just_random_from_top8_entropy_min = mean_top_1_acc_top_n_entropy(8, min_ent=True, skip_top1=True)
        just_random_from_top16_entropy_min = mean_top_1_acc_top_n_entropy(16, min_ent=True, skip_top1=True)

        just_random_from_top1_stderr = mean_top_1_acc_top_n_stderr(1, skip_top1=True)
        just_random_from_top2_stderr = mean_top_1_acc_top_n_stderr(2, skip_top1=True)
        just_random_from_top4_stderr = mean_top_1_acc_top_n_stderr(4, skip_top1=True)
        just_random_from_top8_stderr = mean_top_1_acc_top_n_stderr(8, skip_top1=True)
        just_random_from_top16_stderr = mean_top_1_acc_top_n_stderr(16, skip_top1=True)

        just_random_from_top1_stderr_min = mean_top_1_acc_top_n_stderr(1, min_ent=True, skip_top1=True)
        just_random_from_top2_stderr_min = mean_top_1_acc_top_n_stderr(2, min_ent=True, skip_top1=True)
        just_random_from_top4_stderr_min = mean_top_1_acc_top_n_stderr(4, min_ent=True, skip_top1=True)
        just_random_from_top8_stderr_min = mean_top_1_acc_top_n_stderr(8, min_ent=True, skip_top1=True)
        just_random_from_top16_stderr_min = mean_top_1_acc_top_n_stderr(16, min_ent=True, skip_top1=True)

        just_random_from_top1_kl = mean_top_1_acc_top_n_kl(1, skip_top1=True)
        just_random_from_top2_kl = mean_top_1_acc_top_n_kl(2, skip_top1=True)
        just_random_from_top4_kl = mean_top_1_acc_top_n_kl(4, skip_top1=True)
        just_random_from_top8_kl = mean_top_1_acc_top_n_kl(8, skip_top1=True)
        just_random_from_top16_kl = mean_top_1_acc_top_n_kl(16, skip_top1=True)

        just_random_from_top1_kl_min = mean_top_1_acc_top_n_kl(1, min_ent=True, skip_top1=True)
        just_random_from_top2_kl_min = mean_top_1_acc_top_n_kl(2, min_ent=True, skip_top1=True)
        just_random_from_top4_kl_min = mean_top_1_acc_top_n_kl(4, min_ent=True, skip_top1=True)
        just_random_from_top8_kl_min = mean_top_1_acc_top_n_kl(8, min_ent=True, skip_top1=True)
        just_random_from_top16_kl_min = mean_top_1_acc_top_n_kl(16, min_ent=True, skip_top1=True)


        print(f"Top1: {top1}, Top2: {top2}, Top1_random: {top1_random}")
        print(f"Top1_random_from_top_2: {top1_random_from_top_2}, Top1_random_from_top_4: {top1_random_from_top_4}, Top1_random_from_top_8: {top1_random_from_top_8}, Top1_random_from_top_16: {top1_random_from_top_16}")
        # print(f"Top1hybrid_4_25: {top1andhybrid_4_25}, Top1hybrid_4_50: {top1andhybrid_4_50}, Top1hybrid_4_100: {top1andhybrid_4_100}, Top1hybrid_4_200: {top1andhybrid_4_200}, Top1hybrid_4_400: {top1andhybrid_4_400}, Top1hybrid_4_800: {top1andhybrid_4_800}")
        # print(f"Top1hybrid_8_25: {top1andhybrid_8_25}, Top1hybrid_8_50: {top1andhybrid_8_50}, Top1hybrid_8_100: {top1andhybrid_8_100}, Top1hybrid_8_200: {top1andhybrid_8_200}, Top1hybrid_8_400: {top1andhybrid_8_400}, Top1hybrid_8_800: {top1andhybrid_8_800}")
        # print(f"Top1hybrid_16_25: {top1andhybrid_16_25}, Top1hybrid_16_50: {top1andhybrid_16_50}, Top1hybrid_16_100: {top1andhybrid_16_100}, Top1hybrid_16_200: {top1andhybrid_16_200}, Top1hybrid_16_400: {top1andhybrid_16_400}, Top1hybrid_16_800: {top1andhybrid_16_800}")
        # print(f"Top1hybrid_40_25: {top1andhybrid_40_25}, Top1hybrid_40_50: {top1andhybrid_40_50}, Top1hybrid_40_100: {top1andhybrid_40_100}, Top1hybrid_40_200: {top1andhybrid_40_200}, Top1hybrid_40_400: {top1andhybrid_40_400}, Top1hybrid_40_800: {top1andhybrid_40_800}")
        print(f"Top1_entropy_1: {top1andentropy_1}, Top1_entropy_2: {top1andentropy_2}, Top1_entropy_4: {top1andentropy_4}, Top1_entropy_8: {top1andentropy_8}, Top1_entropy_16: {top1andentropy_16}")
        print(f"Top1_entropy_1_min: {top1andentropy_1_min}, Top1_entropy_2_min: {top1andentropy_2_min}, Top1_entropy_4_min: {top1andentropy_4_min}, Top1_entropy_8_min: {top1andentropy_8_min}, Top1_entropy_16_min: {top1andentropy_16_min}")
        print(f"Top1_std_1: {top1andstd_1}, Top1_std_2: {top1andstd_2}, Top1_std_4: {top1andstd_4}, Top1_std_8: {top1andstd_8}, Top1_std_16: {top1andstd_16}")
        print(f"Top1_std_1_min: {top1andstd_1}, Top1_std_2_min: {top1andstd_2}, Top1_std_4_min: {top1andstd_4}, Top1_std_8_min: {top1andstd_8}, Top1_std_16_min: {top1andstd_16}")
        print(f"Top1_stderr_1: {top1andstderr_1}, Top1_stderr_2: {top1andstderr_2}, Top1_stderr_4: {top1andstderr_4}, Top1_stderr_8: {top1andstderr_8}, Top1_stderr_16: {top1andstderr_16}")
        print(f"Top1_stderr_1_min: {top1andstderr_1_min}, Top1_stderr_2_min: {top1andstderr_2_min}, Top1_stderr_4_min: {top1andstderr_4_min}, Top1_stderr_8_min: {top1andstderr_8_min}, Top1_stderr_16_min: {top1andstderr_16_min}")
        print(f"Top1_kl_1: {top1andkl_1}, Top1_kl_2: {top1andkl_2}, Top1_kl_4: {top1andkl_4}, Top1_kl_8: {top1andkl_8}, Top1_kl_16: {top1andkl_16}")
        print(f"Top1_kl_1_min: {top1andkl_1_min}, Top1_kl_2_min: {top1andkl_2_min}, Top1_kl_4_min: {top1andkl_4_min}, Top1_kl_8_min: {top1andkl_8_min}, Top1_kl_16_min: {top1andkl_16_min}")
        top1s.append(top1)
        top2s.append(top2)
        top1s_random.append(top1_random)
        top1s_random_from_top_2.append(top1_random_from_top_2)
        top1s_random_from_top_4.append(top1_random_from_top_4)
        top1s_random_from_top_8.append(top1_random_from_top_8)
        top1s_random_from_top_16.append(top1_random_from_top_16)
        top1s_random_from_top_40.append(top1_random_from_top_40)
        top1ent1s.append(top1andentropy_1)
        top1ent2s.append(top1andentropy_2)
        top1ent4s.append(top1andentropy_4)
        top1ent8s.append(top1andentropy_8)
        top1ent16s.append(top1andentropy_16)
        top1ent40s.append(top1andentropy_40)
        top1ent1s_min.append(top1andentropy_1_min)
        top1ent2s_min.append(top1andentropy_2_min)
        top1ent4s_min.append(top1andentropy_4_min)
        top1ent8s_min.append(top1andentropy_8_min)
        top1ent16s_min.append(top1andentropy_16_min)
        top1ent40s_min.append(top1andentropy_40_min)    
        top1std1s.append(top1andstd_1)
        top1std2s.append(top1andstd_2)
        top1std4s.append(top1andstd_4)
        top1std8s.append(top1andstd_8)
        top1std16s.append(top1andstd_16)
        top1std40s.append(top1andstd_40)
        top1std1s_min.append(top1andstd_1_min)
        top1std2s_min.append(top1andstd_2_min)
        top1std4s_min.append(top1andstd_4_min)
        top1std8s_min.append(top1andstd_8_min)
        top1std16s_min.append(top1andstd_16_min)
        top1std40s_min.append(top1andstd_40_min)
        top1stderr1s.append(top1andstderr_1)
        top1stderr2s.append(top1andstderr_2)
        top1stderr4s.append(top1andstderr_4)
        top1stderr8s.append(top1andstderr_8)
        top1stderr16s.append(top1andstderr_16)
        top1stderr40s.append(top1andstderr_40)
        top1stderr1s_min.append(top1andstderr_1_min)
        top1stderr2s_min.append(top1andstderr_2_min)
        top1stderr4s_min.append(top1andstderr_4_min)
        top1stderr8s_min.append(top1andstderr_8_min)
        top1stderr16s_min.append(top1andstderr_16_min)
        top1stderr40s_min.append(top1andstderr_40_min)
        top1andkl1s.append(top1andkl_1)
        top1andkl2s.append(top1andkl_2)
        top1andkl4s.append(top1andkl_4)
        top1andkl8s.append(top1andkl_8)
        top1andkl16s.append(top1andkl_16)
        top1andkl40s.append(top1andkl_40)
        top1andkl1s_min.append(top1andkl_1_min)
        top1andkl2s_min.append(top1andkl_2_min)
        top1andkl4s_min.append(top1andkl_4_min)
        top1andkl8s_min.append(top1andkl_8_min)
        top1andkl16s_min.append(top1andkl_16_min)
        top1andkl40s_min.append(top1andkl_40_min)
        justs_random.append(just_random)
        justs_random_from_top1_confidence.append(just_random_from_top1_confidence)
        justs_random_from_top2_confidence.append(just_random_from_top2_confidence)
        justs_random_from_top4_confidence.append(just_random_from_top4_confidence)
        justs_random_from_top8_confidence.append(just_random_from_top8_confidence)
        justs_random_from_top16_confidence.append(just_random_from_top16_confidence)
        justs_random_from_top1_entropy.append(just_random_from_top1_entropy)
        justs_random_from_top2_entropy.append(just_random_from_top2_entropy)
        justs_random_from_top4_entropy.append(just_random_from_top4_entropy)
        justs_random_from_top8_entropy.append(just_random_from_top8_entropy)
        justs_random_from_top16_entropy.append(just_random_from_top16_entropy)
        justs_random_from_top1_entropy_min.append(just_random_from_top1_entropy_min)
        justs_random_from_top2_entropy_min.append(just_random_from_top2_entropy_min)
        justs_random_from_top4_entropy_min.append(just_random_from_top4_entropy_min)
        justs_random_from_top8_entropy_min.append(just_random_from_top8_entropy_min)
        justs_random_from_top16_entropy_min.append(just_random_from_top16_entropy_min)
        justs_random_from_top1_stderr.append(just_random_from_top1_stderr)
        justs_random_from_top2_stderr.append(just_random_from_top2_stderr)
        justs_random_from_top4_stderr.append(just_random_from_top4_stderr)
        justs_random_from_top8_stderr.append(just_random_from_top8_stderr)
        justs_random_from_top16_stderr.append(just_random_from_top16_stderr)
        justs_random_from_top1_stderr_min.append(just_random_from_top1_stderr_min)
        justs_random_from_top2_stderr_min.append(just_random_from_top2_stderr_min)
        justs_random_from_top4_stderr_min.append(just_random_from_top4_stderr_min)
        justs_random_from_top8_stderr_min.append(just_random_from_top8_stderr_min)
        justs_random_from_top16_stderr_min.append(just_random_from_top16_stderr_min)
        justs_random_from_top1_kl.append(just_random_from_top1_kl)
        justs_random_from_top2_kl.append(just_random_from_top2_kl)
        justs_random_from_top4_kl.append(just_random_from_top4_kl)
        justs_random_from_top8_kl.append(just_random_from_top8_kl)
        justs_random_from_top16_kl.append(just_random_from_top16_kl)
        justs_random_from_top1_kl_min.append(just_random_from_top1_kl_min)
        justs_random_from_top2_kl_min.append(just_random_from_top2_kl_min)
        justs_random_from_top4_kl_min.append(just_random_from_top4_kl_min)
        justs_random_from_top8_kl_min.append(just_random_from_top8_kl_min)
        justs_random_from_top16_kl_min.append(just_random_from_top16_kl_min)




print(f"evaluated_rooms: {len(top1s)}")
print(f"evaluated_instances: {evaluated_instances}")


# calculate mean top-k acc
top1hybrid_4_25 = np.mean(top1hybrid_4_25s)
top1 = np.mean(top1s)
top2 = np.mean(top2s)
top1_random = np.mean(top1s_random)
top1_random_from_top_2 = np.mean(top1s_random_from_top_2)
top1_random_from_top_4 = np.mean(top1s_random_from_top_4)
top1_random_from_top_8 = np.mean(top1s_random_from_top_8)
top1_random_from_top_16 = np.mean(top1s_random_from_top_16)
top1_random_from_top_40 = np.mean(top1s_random_from_top_40)
top1_entropy = np.mean(top1ent1s)
top2_entropy = np.mean(top1ent2s)
top4_entropy = np.mean(top1ent4s)
top8_entropy = np.mean(top1ent8s)
top16_entropy = np.mean(top1ent16s)
top40_entropy = np.mean(top1ent40s)
top1_entropy_min = np.mean(top1ent1s_min)
top2_entropy_min = np.mean(top1ent2s_min)
top4_entropy_min = np.mean(top1ent4s_min)
top8_entropy_min = np.mean(top1ent8s_min)
top16_entropy_min = np.mean(top1ent16s_min)
top40_entropy_min = np.mean(top1ent40s_min)
top1_std = np.mean(top1std1s)
top2_std = np.mean(top1std2s)
top4_std = np.mean(top1std4s)
top8_std = np.mean(top1std8s)
top16_std = np.mean(top1std16s)
top40_std = np.mean(top1std40s)
top1_std_min = np.mean(top1std1s_min)
top2_std_min = np.mean(top1std2s_min)
top4_std_min = np.mean(top1std4s_min)
top8_std_min = np.mean(top1std8s_min)
top16_std_min = np.mean(top1std16s_min)
top40_std_min = np.mean(top1std40s_min)
top1_stderr = np.mean(top1stderr1s)
top2_stderr = np.mean(top1stderr2s)
top4_stderr = np.mean(top1stderr4s)
top8_stderr = np.mean(top1stderr8s)
top16_stderr = np.mean(top1stderr16s)
top40_stderr = np.mean(top1stderr40s)
top1_stderr_min = np.mean(top1stderr1s_min)
top2_stderr_min = np.mean(top1stderr2s_min)
top4_stderr_min = np.mean(top1stderr4s_min)
top8_stderr_min = np.mean(top1stderr8s_min)
top16_stderr_min = np.mean(top1stderr16s_min)
top40_stderr_min = np.mean(top1stderr40s_min)
top1andkl1 = np.mean(top1andkl1s)
top1andkl2 = np.mean(top1andkl2s)
top1andkl4 = np.mean(top1andkl4s)
top1andkl8 = np.mean(top1andkl8s)
top1andkl16 = np.mean(top1andkl16s)
top1andkl40 = np.mean(top1andkl40s)
top1andkl1_min = np.mean(top1andkl1s_min)
top1andkl2_min = np.mean(top1andkl2s_min)
top1andkl4_min = np.mean(top1andkl4s_min)
top1andkl8_min = np.mean(top1andkl8s_min)
top1andkl16_min = np.mean(top1andkl16s_min)
top1andkl40_min = np.mean(top1andkl40s_min)
just_random = np.mean(justs_random)
just_random_from_top1_confidence = np.mean(justs_random_from_top1_confidence)
just_random_from_top2_confidence = np.mean(justs_random_from_top2_confidence)
just_random_from_top4_confidence = np.mean(justs_random_from_top4_confidence)
just_random_from_top8_confidence = np.mean(justs_random_from_top8_confidence)
just_random_from_top16_confidence = np.mean(justs_random_from_top16_confidence)
just_random_from_top1_entropy = np.mean(justs_random_from_top1_entropy)
just_random_from_top2_entropy = np.mean(justs_random_from_top2_entropy)
just_random_from_top4_entropy = np.mean(justs_random_from_top4_entropy)
just_random_from_top8_entropy = np.mean(justs_random_from_top8_entropy)
just_random_from_top16_entropy = np.mean(justs_random_from_top16_entropy)
just_random_from_top1_entropy_min = np.mean(justs_random_from_top1_entropy_min)
just_random_from_top2_entropy_min = np.mean(justs_random_from_top2_entropy_min)
just_random_from_top4_entropy_min = np.mean(justs_random_from_top4_entropy_min)
just_random_from_top8_entropy_min = np.mean(justs_random_from_top8_entropy_min)
just_random_from_top16_entropy_min = np.mean(justs_random_from_top16_entropy_min)
just_random_from_top1_stderr = np.mean(justs_random_from_top1_stderr)
just_random_from_top2_stderr = np.mean(justs_random_from_top2_stderr)
just_random_from_top4_stderr = np.mean(justs_random_from_top4_stderr)
just_random_from_top8_stderr = np.mean(justs_random_from_top8_stderr)
just_random_from_top16_stderr = np.mean(justs_random_from_top16_stderr)
just_random_from_top1_stderr_min = np.mean(justs_random_from_top1_stderr_min)
just_random_from_top2_stderr_min = np.mean(justs_random_from_top2_stderr_min)
just_random_from_top4_stderr_min = np.mean(justs_random_from_top4_stderr_min)
just_random_from_top8_stderr_min = np.mean(justs_random_from_top8_stderr_min)
just_random_from_top16_stderr_min = np.mean(justs_random_from_top16_stderr_min)
just_random_from_top1_kl = np.mean(justs_random_from_top1_kl)
just_random_from_top2_kl = np.mean(justs_random_from_top2_kl)
just_random_from_top4_kl = np.mean(justs_random_from_top4_kl)
just_random_from_top8_kl = np.mean(justs_random_from_top8_kl)
just_random_from_top16_kl = np.mean(justs_random_from_top16_kl)
just_random_from_top1_kl_min = np.mean(justs_random_from_top1_kl_min)
just_random_from_top2_kl_min = np.mean(justs_random_from_top2_kl_min)
just_random_from_top4_kl_min = np.mean(justs_random_from_top4_kl_min)
just_random_from_top8_kl_min = np.mean(justs_random_from_top8_kl_min)
just_random_from_top16_kl_min = np.mean(justs_random_from_top16_kl_min)

s_top1_stderr = np.std(top1s) / np.sqrt(len(top1s))
s_top2_stderr = np.std(top2s) / np.sqrt(len(top2s))
s_top1_random_stderr = np.std(top1s_random) / np.sqrt(len(top1s_random))
s_top1_random_from_top_2_stderr = np.std(top1s_random_from_top_2) / np.sqrt(len(top1s_random_from_top_2))
s_top1_random_from_top_4_stderr = np.std(top1s_random_from_top_4) / np.sqrt(len(top1s_random_from_top_4))
s_top1_random_from_top_8_stderr = np.std(top1s_random_from_top_8) / np.sqrt(len(top1s_random_from_top_8))
s_top1_random_from_top_16_stderr = np.std(top1s_random_from_top_16) / np.sqrt(len(top1s_random_from_top_16))
s_top1_entropy_stderr = np.std(top1ent1s) / np.sqrt(len(top1ent1s))
s_top2_entropy_stderr = np.std(top1ent2s) / np.sqrt(len(top1ent2s))
s_top4_entropy_stderr = np.std(top1ent4s) / np.sqrt(len(top1ent4s))
s_top8_entropy_stderr = np.std(top1ent8s) / np.sqrt(len(top1ent8s))
s_top16_entropy_stderr = np.std(top1ent16s) / np.sqrt(len(top1ent16s))
s_top1_entropy_min_stderr = np.std(top1ent1s_min) / np.sqrt(len(top1ent1s_min))
s_top2_entropy_min_stderr = np.std(top1ent2s_min) / np.sqrt(len(top1ent2s_min))
s_top4_entropy_min_stderr = np.std(top1ent4s_min) / np.sqrt(len(top1ent4s_min))
s_top8_entropy_min_stderr = np.std(top1ent8s_min) / np.sqrt(len(top1ent8s_min))
s_top16_entropy_min_stderr = np.std(top1ent16s_min) / np.sqrt(len(top1ent16s_min))
s_top1_std_stderr = np.std(top1std1s) / np.sqrt(len(top1std1s))
s_top2_std_stderr = np.std(top1std2s) / np.sqrt(len(top1std2s))
s_top4_std_stderr = np.std(top1std4s) / np.sqrt(len(top1std4s))
s_top8_std_stderr = np.std(top1std8s) / np.sqrt(len(top1std8s))
s_top16_std_stderr = np.std(top1std16s) / np.sqrt(len(top1std16s))
s_top1_std_min_stderr = np.std(top1std1s_min) / np.sqrt(len(top1std1s_min))
s_top2_std_min_stderr = np.std(top1std2s_min) / np.sqrt(len(top1std2s_min))
s_top4_std_min_stderr = np.std(top1std4s_min) / np.sqrt(len(top1std4s_min))
s_top8_std_min_stderr = np.std(top1std8s_min) / np.sqrt(len(top1std8s_min))
s_top16_std_min_stderr = np.std(top1std16s_min) / np.sqrt(len(top1std16s_min))
s_top1_stderr_stderr = np.std(top1stderr1s) / np.sqrt(len(top1stderr1s))
s_top2_stderr_stderr = np.std(top1stderr2s) / np.sqrt(len(top1stderr2s))
s_top4_stderr_stderr = np.std(top1stderr4s) / np.sqrt(len(top1stderr4s))
s_top8_stderr_stderr = np.std(top1stderr8s) / np.sqrt(len(top1stderr8s))
s_top16_stderr_stderr = np.std(top1stderr16s) / np.sqrt(len(top1stderr16s))
s_top1_stderr_min_stderr = np.std(top1stderr1s_min) / np.sqrt(len(top1stderr1s_min))
s_top2_stderr_min_stderr = np.std(top1stderr2s_min) / np.sqrt(len(top1stderr2s_min))
s_top4_stderr_min_stderr = np.std(top1stderr4s_min) / np.sqrt(len(top1stderr4s_min))
s_top8_stderr_min_stderr = np.std(top1stderr8s_min) / np.sqrt(len(top1stderr8s_min))
s_top16_stderr_min_stderr = np.std(top1stderr16s_min) / np.sqrt(len(top1stderr16s_min))
s_top1_kl_stderr = np.std(top1andkl1s) / np.sqrt(len(top1andkl1s))
s_top2_kl_stderr = np.std(top1andkl2s) / np.sqrt(len(top1andkl2s))
s_top4_kl_stderr = np.std(top1andkl4s) / np.sqrt(len(top1andkl4s))
s_top8_kl_stderr = np.std(top1andkl8s) / np.sqrt(len(top1andkl8s))
s_top16_kl_stderr = np.std(top1andkl16s) / np.sqrt(len(top1andkl16s))
s_top1_kl_min_stderr = np.std(top1andkl1s_min) / np.sqrt(len(top1andkl1s_min))
s_top2_kl_min_stderr = np.std(top1andkl2s_min) / np.sqrt(len(top1andkl2s_min))
s_top4_kl_min_stderr = np.std(top1andkl4s_min) / np.sqrt(len(top1andkl4s_min))
s_top8_kl_min_stderr = np.std(top1andkl8s_min) / np.sqrt(len(top1andkl8s_min))
s_top16_kl_min_stderr = np.std(top1andkl16s_min) / np.sqrt(len(top1andkl16s_min))

### save the raw data
import json
with open(f"time_raw_confidence.json", "w") as f:
    json.dump(time_recordings, f, indent=4)

time_recordings_mean = {}
time_recordings_std = {}
for k in time_recordings:
    time_recordings_mean[k] = np.mean(time_recordings[k])
    time_recordings_std[k] = np.std(time_recordings[k])

# vis time as line chart
# [top1andentropy_{k}s, top1andstderr{k}s, top1andkl_{k}s]
# k=[1,2,4,8,16]
# only catecories without min
import matplotlib.pyplot as plt
import numpy as np

# data
x = np.array([1, 2, 4, 8, 16])
y1 = np.array([time_recordings_mean[f"top1andentropy_{k}s"] for k in x])
y2 = np.array([time_recordings_mean[f"top1andstderr_{k}s"] for k in x])
y3 = np.array([time_recordings_mean[f"top1andkl_{k}s"] for k in x])
# std
s1 = np.array([time_recordings_std[f"top1andentropy_{k}s"] for k in x])
s2 = np.array([time_recordings_std[f"top1andstderr_{k}s"] for k in x])
s3 = np.array([time_recordings_std[f"top1andkl_{k}s"] for k in x])

# plot
fig, ax = plt.subplots()
ax.errorbar(x, y1, yerr=s1, fmt='-o', label="entropy", alpha=0.7)   
ax.errorbar(x, y2, yerr=s2, fmt='-o', label="stderr", alpha=0.7)
ax.errorbar(x, y3, yerr=s3, fmt='-o', label="kl", alpha=0.7)
ax.set_xlabel("k")
ax.set_ylabel("time (s)")
plt.legend()

# save
plt.savefig("time_line_chart_conf.png")


# save results
with open(f"time_top_conf.json", "w") as f:
    json.dump(time_recordings, f, indent=4)


print(f"Top1: {top1}, Top2: {top2}, Top1_random: {top1_random}")
print(f"Top1_random_from_top_2: {top1_random_from_top_2}, Top1_random_from_top_4: {top1_random_from_top_4}, Top1_random_from_top_8: {top1_random_from_top_8}, Top1_random_from_top_16: {top1_random_from_top_16}")
# print(f"Top1hybrid_4_25: {top1hybrid_4_25}, Top1hybrid_4_50: {top1hybrid_4_50}, Top1hybrid_4_100: {top1hybrid_4_100}, Top1hybrid_4_200: {top1hybrid_4_200}, Top1hybrid_4_400: {top1hybrid_4_400}, Top1hybrid_4_800: {top1hybrid_4_800}")
# print(f"Top1hybrid_8_25: {top1hybrid_8_25}, Top1hybrid_8_50: {top1hybrid_8_50}, Top1hybrid_8_100: {top1hybrid_8_100}, Top1hybrid_8_200: {top1hybrid_8_200}, Top1hybrid_8_400: {top1hybrid_8_400}, Top1hybrid_8_800: {top1hybrid_8_800}")
# print(f"Top1hybrid_16_25: {top1hybrid_16_25}, Top1hybrid_16_50: {top1hybrid_16_50}, Top1hybrid_16_100: {top1hybrid_16_100}, Top1hybrid_16_200: {top1hybrid_16_200}, Top1hybrid_16_400: {top1hybrid_16_400}, Top1hybrid_16_800: {top1hybrid_16_800}")
# print(f"Top1hybrid_40_25: {top1hybrid_40_25}, Top1hybrid_40_50: {top1hybrid_40_50}, Top1hybrid_40_100: {top1hybrid_40_100}, Top1hybrid_40_200: {top1hybrid_40_200}, Top1hybrid_40_400: {top1hybrid_40_400}, Top1hybrid_40_800: {top1hybrid_40_800}")
print(f"Top1_entropy: {top1_entropy}, Top2_entropy: {top2_entropy}, Top4_entropy: {top4_entropy}, Top8_entropy: {top8_entropy}, Top16_entropy: {top16_entropy}")
print(f"Top1_entropy_min: {top1_entropy_min}, Top2_entropy_min: {top2_entropy_min}, Top4_entropy_min: {top4_entropy_min}, Top8_entropy_min: {top8_entropy_min}, Top16_entropy_min: {top16_entropy_min}")
print(f"Top1_std: {top1_std}, Top2_std: {top2_std}, Top4_std: {top4_std}, Top8_std: {top8_std}, Top16_std: {top16_std}")
print(f"Top1_std_min: {top1_std_min}, Top2_std_min: {top2_std_min}, Top4_std_min: {top4_std_min}, Top8_std_min: {top8_std_min}, Top16_std_min: {top16_std_min}")
print(f"Top1_stderr: {top1_stderr}, Top2_stderr: {top2_stderr}, Top4_stderr: {top4_stderr}, Top8_stderr: {top8_stderr}, Top16_stderr: {top16_stderr}")
print(f"Top1_stderr_min: {top1_stderr_min}, Top2_stderr_min: {top2_stderr_min}, Top4_stderr_min: {top4_stderr_min}, Top8_stderr_min: {top8_stderr_min}, Top16_stderr_min: {top16_stderr_min}")
print(f"Top1_kl: {top1andkl1}, Top2_kl: {top1andkl2}, Top4_kl: {top1andkl4}, Top8_kl: {top1andkl8}, Top16_kl: {top1andkl16}")
print(f"Top1_kl_min: {top1andkl1_min}, Top2_kl_min: {top1andkl2_min}, Top4_kl_min: {top1andkl4_min}, Top8_kl_min: {top1andkl8_min}, Top16_kl_min: {top1andkl16_min}")

# print stderr
print(f"Top1: {s_top1_stderr}, Top2: {s_top2_stderr}, Top1_random: {s_top1_random_stderr}")
print(f"Top1_random_from_top_2: {s_top1_random_from_top_2_stderr}, Top1_random_from_top_4: {s_top1_random_from_top_4_stderr}, Top1_random_from_top_8: {s_top1_random_from_top_8_stderr}, Top1_random_from_top_16: {s_top1_random_from_top_16_stderr}")
# print(f"Top1hybrid_4_25: {s_top1_hybrid_4_25}, Top1hybrid_4_50: {s_top1_hybrid_4_50}, Top1hybrid_4_100: {s_top1_hybrid_4_100}, Top1hybrid_4_200: {s_top1_hybrid_4_200}, Top1hybrid_4_400: {s_top1_hybrid_4_400}, Top1hybrid_4_800: {s_top1_hybrid_4_800}")
# print(f"Top1hybrid_8_25: {s_top1_hybrid_8_25}, Top1hybrid_8_50: {s_top1_hybrid_8_50}, Top1hybrid_8_100: {s_top1_hybrid_8_100}, Top1hybrid_8_200: {s_top1_hybrid_8_200}, Top1hybrid_8_400: {s_top1_hybrid_8_400}, Top1hybrid_8_800: {s_top1_hybrid_8_800}")
# print(f"Top1hybrid_16_25: {s_top1_hybrid_16_25}, Top1hybrid_16_50: {s_top1_hybrid_16_50}, Top1hybrid_16_100: {s_top1_hybrid_16_100}, Top1hybrid_16_200: {s_top1_hybrid_16_200}, Top1hybrid_16_400: {s_top1_hybrid_16_400}, Top1hybrid_16_800: {s_top1_hybrid_16_800}")
# print(f"Top1hybrid_40_25: {s_top1_hybrid_40_25}, Top1hybrid_40_50: {s_top1_hybrid_40_50}, Top1hybrid_40_100: {s_top1_hybrid_40_100}, Top1hybrid_40_200: {s_top1_hybrid_40_200}, Top1hybrid_40_400: {s_top1_hybrid_40_400}, Top1hybrid_40_800: {s_top1_hybrid_40_800}")
print(f"Top1_entropy: {s_top1_entropy_stderr}, Top2_entropy: {s_top2_entropy_stderr}, Top4_entropy: {s_top4_entropy_stderr}, Top8_entropy: {s_top8_entropy_stderr}, Top16_entropy: {s_top16_entropy_stderr}")
print(f"Top1_entropy_min: {s_top1_entropy_min_stderr}, Top2_entropy_min: {s_top2_entropy_min_stderr}, Top4_entropy_min: {s_top4_entropy_min_stderr}, Top8_entropy_min: {s_top8_entropy_min_stderr}, Top16_entropy_min: {s_top16_entropy_min_stderr}")
print(f"Top1_std: {s_top1_std_stderr}, Top2_std: {s_top2_std_stderr}, Top4_std: {s_top4_std_stderr}, Top8_std: {s_top8_std_stderr}, Top16_std: {s_top16_std_stderr}")
print(f"Top1_std_min: {s_top1_std_min_stderr}, Top2_std_min: {s_top2_std_min_stderr}, Top4_std_min: {s_top4_std_min_stderr}, Top8_std_min: {s_top8_std_min_stderr}, Top16_std_min: {s_top16_std_min_stderr}")
print(f"Top1_stderr: {s_top1_stderr_stderr}, Top2_stderr: {s_top2_stderr_stderr}, Top4_stderr: {s_top4_stderr_stderr}, Top8_stderr: {s_top8_stderr_stderr}, Top16_stderr: {s_top16_stderr_stderr}")
print(f"Top1_stderr_min: {s_top1_stderr_min_stderr}, Top2_stderr_min: {s_top2_stderr_min_stderr}, Top4_stderr_min: {s_top4_stderr_min_stderr}, Top8_stderr_min: {s_top8_stderr_min_stderr}, Top16_stderr_min: {s_top16_stderr_min_stderr}")
print(f"Top1_kl: {s_top1_kl_stderr}, Top2_kl: {s_top2_kl_stderr}, Top4_kl: {s_top4_kl_stderr}, Top8_kl: {s_top8_kl_stderr}, Top16_kl: {s_top16_kl_stderr}")
print(f"Top1_kl_min: {s_top1_kl_min_stderr}, Top2_kl_min: {s_top2_kl_min_stderr}, Top4_kl_min: {s_top4_kl_min_stderr}, Top8_kl_min: {s_top8_kl_min_stderr}, Top16_kl_min: {s_top16_kl_min_stderr}")

# save as json
# entries: min_entropy, max_kl, max_stderr, just_random, just_random_from_topk_confidence, just...
# Ks: 1, 2, 4, 8, 16
output = {
    "top1": top1,
    "top1+random": top1_random,
    "top2": top2,
    "top1+random_from_topk_confidence": {
        "1": top2,
        "2": top1_random_from_top_2,
        "4": top1_random_from_top_4,
        "8": top1_random_from_top_8,
        "16": top1_random_from_top_16,
        "40": top1_random_from_top_40
    },
    "top1+min_entropy_from_topk_confidence": {
        "1": top1_entropy_min,
        "2": top2_entropy_min,
        "4": top4_entropy_min,
        "8": top8_entropy_min,
        "16": top16_entropy_min,
        "40": top40_entropy_min
    },
    "top1+max_entropy_from_topk_confidence": {
        "1": top1_entropy,
        "2": top2_entropy,
        "4": top4_entropy,
        "8": top8_entropy,
        "16": top16_entropy,
        "40": top40_entropy
    },
    "top1+min_kl_from_topk_confidence": {
        "1": top1andkl1_min,
        "2": top1andkl2_min,
        "4": top1andkl4_min,
        "8": top1andkl8_min,
        "16": top1andkl16_min,
        "40": top1andkl40_min
    },
    "top1+max_kl_from_topk_confidence": {
        "1": top1andkl1,
        "2": top1andkl2,
        "4": top1andkl4,
        "8": top1andkl8,
        "16": top1andkl16,
        "40": top1andkl40
    },
    "top1+min_stderr_from_topk_confidence": {
        "1": top1_stderr_min,
        "2": top2_stderr_min,
        "4": top4_stderr_min,
        "8": top8_stderr_min,
        "16": top16_stderr_min,
        "40": top40_stderr_min
    },
    "top1+max_stderr_from_topk_confidence": {
        "1": top1_stderr,
        "2": top2_stderr,
        "4": top4_stderr,
        "8": top8_stderr,
        "16": top16_stderr,
        "40": top40_stderr  
    },
    "just_random": just_random, 
    "just_random_from_topk_confidence": {
        "1": just_random_from_top1_confidence,
        "2": just_random_from_top2_confidence,
        "4": just_random_from_top4_confidence,
        "8": just_random_from_top8_confidence,
        "16": just_random_from_top16_confidence
    },
    "just_max_entropy_from_topk_confidence": {
        "1": just_random_from_top1_entropy,
        "2": just_random_from_top2_entropy,
        "4": just_random_from_top4_entropy,
        "8": just_random_from_top8_entropy,
        "16": just_random_from_top16_entropy
    },
    "just_min_entropy_from_topk_confidence": {
        "1": just_random_from_top1_entropy_min,
        "2": just_random_from_top2_entropy_min,
        "4": just_random_from_top4_entropy_min,
        "8": just_random_from_top8_entropy_min,
        "16": just_random_from_top16_entropy_min
    },
    "just_max_stderr_from_topk_confidence": {
        "1": just_random_from_top1_stderr,
        "2": just_random_from_top2_stderr,
        "4": just_random_from_top4_stderr,
        "8": just_random_from_top8_stderr,
        "16": just_random_from_top16_stderr
    },
    "just_min_stderr_from_topk_confidence": {
        "1": just_random_from_top1_stderr_min,
        "2": just_random_from_top2_stderr_min,
        "4": just_random_from_top4_stderr_min,
        "8": just_random_from_top8_stderr_min,
        "16": just_random_from_top16_stderr_min
    },
    "just_max_kl_from_topk_confidence": {
        "1": just_random_from_top1_kl,
        "2": just_random_from_top2_kl,
        "4": just_random_from_top4_kl,
        "8": just_random_from_top8_kl,
        "16": just_random_from_top16_kl
    },
    "just_min_kl_from_topk_confidence": {
        "1": just_random_from_top1_kl_min,
        "2": just_random_from_top2_kl_min,
        "4": just_random_from_top4_kl_min,
        "8": just_random_from_top8_kl_min,
        "16": just_random_from_top16_kl_min
    }
} 

with open("topk_confidence_replanning.json", "w") as f:
    json.dump(output, f, indent=4)


# also store raw data
output_raw = {
    "top1": top1s,
    "top2": top2s,
    "top1_random": top1s_random,
    "top1+random_from_top_k_confidence": {
        "1": top2s,
        "2": top1s_random_from_top_2,
        "4": top1s_random_from_top_4,
        "8": top1s_random_from_top_8,
        "16": top1s_random_from_top_16
    },
    "top1+max_entropy_from_top_k_confidence": {
        "1": top1ent1s,
        "2": top1ent2s,
        "4": top1ent4s,
        "8": top1ent8s,
        "16": top1ent16s
    },
    "top1+min_entropy_from_top_k_confidence": {
        "1": top1ent1s_min,
        "2": top1ent2s_min,
        "4": top1ent4s_min,
        "8": top1ent8s_min,
        "16": top1ent16s_min
    },
    "top1+max_stderr_from_top_k_confidence": {
        "1": top1stderr1s,
        "2": top1stderr2s,
        "4": top1stderr4s,
        "8": top1stderr8s,
        "16": top1stderr16s
    },
    "top1+min_stderr_from_top_k_confidence": {
        "1": top1stderr1s_min,
        "2": top1stderr2s_min,
        "4": top1stderr4s_min,
        "8": top1stderr8s_min,
        "16": top1stderr16s_min
    },
    "top1+max_kl_from_top_k_confidence": {
        "1": top1andkl1s,
        "2": top1andkl2s,
        "4": top1andkl4s,
        "8": top1andkl8s,
        "16": top1andkl16s
    },
    "top1+min_kl_from_top_k_confidence": {
        "1": top1andkl1s_min,
        "2": top1andkl2s_min,
        "4": top1andkl4s_min,
        "8": top1andkl8s_min,
        "16": top1andkl16s_min
    },
    "just_random": justs_random,
    "just_random_from_top_k_confidence": {
        "1": justs_random_from_top1_confidence,
        "2": justs_random_from_top2_confidence,
        "4": justs_random_from_top4_confidence,
        "8": justs_random_from_top8_confidence,
        "16": justs_random_from_top16_confidence
    },
    "just_max_entropy_from_top_k_confidence": {
        "1": justs_random_from_top1_entropy,
        "2": justs_random_from_top2_entropy,
        "4": justs_random_from_top4_entropy,
        "8": justs_random_from_top8_entropy,
        "16": justs_random_from_top16_entropy
    },
    "just_min_entropy_from_top_k_confidence": {
        "1": justs_random_from_top1_entropy_min,
        "2": justs_random_from_top2_entropy_min,
        "4": justs_random_from_top4_entropy_min,
        "8": justs_random_from_top8_entropy_min,
        "16": justs_random_from_top16_entropy_min
    },
    "just_max_stderr_from_top_k_confidence": {
        "1": justs_random_from_top1_stderr,
        "2": justs_random_from_top2_stderr,
        "4": justs_random_from_top4_stderr,
        "8": justs_random_from_top8_stderr,
        "16": justs_random_from_top16_stderr
    },
    "just_min_stderr_from_top_k_confidence": {
        "1": justs_random_from_top1_stderr_min,
        "2": justs_random_from_top2_stderr_min,
        "4": justs_random_from_top4_stderr_min,
        "8": justs_random_from_top8_stderr_min,
        "16": justs_random_from_top16_stderr_min
    },
    "just_max_kl_from_top_k_confidence": {
        "1": justs_random_from_top1_kl,
        "2": justs_random_from_top2_kl,
        "4": justs_random_from_top4_kl,
        "8": justs_random_from_top8_kl,
        "16": justs_random_from_top16_kl
    },
    "just_min_kl_from_top_k_confidence": {
        "1": justs_random_from_top1_kl_min,
        "2": justs_random_from_top2_kl_min,
        "4": justs_random_from_top4_kl_min,
        "8": justs_random_from_top8_kl_min,
        "16": justs_random_from_top16_kl_min
    }
}

with open("topk_confidence_replanning_raw.json", "w") as f:
    json.dump(output_raw, f, indent=4)


import matplotlib.pyplot as plt

# 1-1 performance of entropy/min_entropy/confidence vs k
plt.figure()
plt.errorbar([2, 4, 8, 16], [top2_entropy, top4_entropy, top8_entropy, top16_entropy], yerr=[s_top2_entropy_stderr, s_top4_entropy_stderr, s_top8_entropy_stderr, s_top1_entropy_stderr], label="Max Entropy", fmt='o-', alpha=0.5, capsize=3)
plt.errorbar([2, 4, 8, 16], [top2_entropy_min, top4_entropy_min, top8_entropy_min, top16_entropy_min], yerr=[s_top2_entropy_min_stderr, s_top4_entropy_min_stderr, s_top8_entropy_min_stderr, s_top1_entropy_min_stderr], label="Min Entropy", fmt='o-', alpha=0.5, capsize=3)
# also plot the random baseline and top2 baseline along with their stderr
plt.errorbar([2, 4, 8, 16], [top1_random_from_top_2, top1_random_from_top_4, top1_random_from_top_8, top1_random_from_top_16], yerr=[s_top1_random_from_top_2_stderr, s_top1_random_from_top_4_stderr, s_top1_random_from_top_8_stderr, s_top1_random_from_top_16_stderr], label="Random selection from topk", fmt='o-', alpha=0.5, capsize=3)
# single horizontal line for top1 and top2
plt.axhline(y=top1, color='r', linestyle='--', label="Top1", alpha=0.5)
plt.axhline(y=top2, color='g', linestyle='--', label="Top2", alpha=0.5)
# save figure
plt.xlabel("k")
plt.ylabel("Retrieval accuracy")
plt.legend()
plt.savefig(f"./entropy_min_entropy_confidence_vs_k.png")
plt.close()

# another version without error bars
plt.figure()
plt.plot([2, 4, 8, 16], [top2_entropy, top4_entropy, top8_entropy, top16_entropy], label="Max Entropy", marker='o')
plt.plot([2, 4, 8, 16], [top2_entropy_min, top4_entropy_min, top8_entropy_min, top16_entropy_min], label="Min Entropy", marker='o')
# also plot the random baseline and top2 baseline along with their stderr
plt.plot([2, 4, 8, 16], [top1_random_from_top_2, top1_random_from_top_4, top1_random_from_top_8, top1_random_from_top_16], label="Random selection from topk", marker='o')
# single horizontal line for top1 and top2
plt.axhline(y=top1, color='r', linestyle='--', label="Top1")
plt.axhline(y=top2, color='g', linestyle='--', label="Top2")
# save figure
plt.xlabel("k")
plt.ylabel("Retrieval accuracy")
plt.legend()
plt.savefig(f"./entropy_min_entropy_confidence_vs_k_no_errorbars.png")
plt.close()



# 1-2 same for kl-divergence
plt.figure()
plt.errorbar([2, 4, 8, 16], [top1andkl2, top1andkl4, top1andkl8, top1andkl16], yerr=[s_top2_kl_stderr, s_top4_kl_stderr, s_top8_kl_stderr, s_top16_kl_stderr], label="KL Divergence", fmt='o-', alpha=0.5, capsize=3)
plt.errorbar([2, 4, 8, 16], [top1andkl2_min, top1andkl4_min, top1andkl8_min, top1andkl16_min], yerr=[s_top2_kl_min_stderr, s_top4_kl_min_stderr, s_top8_kl_min_stderr, s_top16_kl_min_stderr], label="Min KL Divergence", fmt='o-', alpha=0.5, capsize=3)
# also plot the random baseline and top2 baseline along with their stderr
plt.errorbar([2, 4, 8, 16], [top1_random_from_top_2, top1_random_from_top_4, top1_random_from_top_8, top1_random_from_top_16], yerr=[s_top1_random_from_top_2_stderr, s_top1_random_from_top_4_stderr, s_top1_random_from_top_8_stderr, s_top1_random_from_top_16_stderr], label="Random selection from topk", fmt='o-', alpha=0.5, capsize=3)
# single horizontal line for top1 and top2
plt.axhline(y=top1, color='r', linestyle='--', label="Top1", alpha=0.5)
plt.axhline(y=top2, color='g', linestyle='--', label="Top2", alpha=0.5)
# save figure
plt.xlabel("k")
plt.ylabel("Retrieval accuracy")
plt.legend()
plt.savefig(f"./kl_divergence_min_kl_divergence_vs_k.png")
plt.close()

# another version without error bars
plt.figure()
plt.plot([2, 4, 8, 16], [top1andkl2, top1andkl4, top1andkl8, top1andkl16], label="KL Divergence", marker='o')
plt.plot([2, 4, 8, 16], [top1andkl2_min, top1andkl4_min, top1andkl8_min, top1andkl16_min], label="Min KL Divergence", marker='o')
# also plot the random baseline and top2 baseline along with their stderr
plt.plot([2, 4, 8, 16], [top1_random_from_top_2, top1_random_from_top_4, top1_random_from_top_8, top1_random_from_top_16], label="Random selection from topk", marker='o')
# single horizontal line for top1 and top2
plt.axhline(y=top1, color='r', linestyle='--', label="Top1")
plt.axhline(y=top2, color='g', linestyle='--', label="Top2")
# save figure
plt.xlabel("k")
plt.ylabel("Retrieval accuracy")
plt.legend()
plt.savefig(f"./kl_divergence_min_kl_divergence_vs_k_no_errorbars.png")
plt.close()




