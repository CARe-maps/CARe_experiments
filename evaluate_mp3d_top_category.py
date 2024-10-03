
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
    mp3d_categoty_mapping = "./matterport_category_mappings.tsv"
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
top1hybrid_4_1600s = []
top1hybrid_4_4000s = []
top1hybrid_4_10000s = []
top1hybrid_4_20000s = []
top1hybrid_8_25s = []
top1hybrid_8_50s = []
top1hybrid_8_100s = []
top1hybrid_8_200s = []
top1hybrid_8_400s = []
top1hybrid_8_800s = []
top1hybrid_8_1600s = []
top1hybrid_8_4000s = []
top1hybrid_8_10000s = []
top1hybrid_8_20000s = []
top1hybrid_16_25s = []
top1hybrid_16_50s = []
top1hybrid_16_100s = []
top1hybrid_16_200s = []
top1hybrid_16_400s = []
top1hybrid_16_800s = []
top1hybrid_16_1600s = []
top1hybrid_16_4000s = []
top1hybrid_16_10000s = []
top1hybrid_16_20000s = []
top1hybrid_40_25s = []
top1hybrid_40_50s = []
top1hybrid_40_100s = []
top1hybrid_40_200s = []
top1hybrid_40_400s = []
top1hybrid_40_800s = []
top1hybrid_40_1600s = []
top1hybrid_40_4000s = []
top1hybrid_40_10000s = []
top1hybrid_40_20000s = []
top1hybrid_100_25s = []
top1hybrid_100_50s = []
top1hybrid_100_100s = []
top1hybrid_100_200s = []
top1hybrid_100_400s = []
top1hybrid_100_800s = []
top1hybrid_100_1600s = []
top1hybrid_100_4000s = []
top1hybrid_100_10000s = []
top1hybrid_100_20000s = []
top1s = []
top2s = []
top1s_random = []
top1s_random_top_2 = []
top1s_random_top_4 = []
top1s_random_top_8 = []
top1s_random_top_16 = []
top1s_random_top_40 = []
top1s_random_top_100 = []
top1ent1s = []
top1ent2s = []
top1ent4s = []
top1ent8s = []
top1ent16s = []
top1ent40s = []
top1ent100s = []
top1ent200s = []
top1ent500s = []
top1ent1s_min = []
top1ent2s_min = []
top1ent4s_min = []
top1ent8s_min = []
top1ent16s_min = []
top1ent40s_min = []
top1ent100s_min = []
top1ent200s_min = []
top1ent500s_min = []
top1std1s = []
top1std2s = []
top1std4s = []
top1std8s = []
top1std16s = []
top1std40s = []
top1std100s = []
top1std200s = []
top1std500s = []
top1stderr1s = []
top1stderr2s = []
top1stderr4s = []
top1stderr8s = []
top1stderr16s = []
top1stderr40s = []
top1stderr100s = []
top1stderr200s = []
top1stderr500s = []
top1std1s_min = []
top1std2s_min = []
top1std4s_min = []
top1std8s_min = []
top1std16s_min = []
top1std40s_min = []
top1std100s_min = []
top1std200s_min = []
top1std500s_min = []
top1stderr1s_min = []
top1stderr2s_min = []
top1stderr4s_min = []
top1stderr8s_min = []
top1stderr16s_min = []
top1stderr40s_min = []
top1stderr100s_min = []
top1stderr200s_min = []
top1stderr500s_min = []
top1andkl1s = []
top1andkl2s = []
top1andkl4s = []
top1andkl8s = []
top1andkl16s = []
top1andkl40s = []
top1andkl100s = []
top1andkl200s = []
top1andkl500s = []
top1andkl1s_min = []
top1andkl2s_min = []
top1andkl4s_min = []
top1andkl8s_min = []
top1andkl16s_min = []
top1andkl40s_min = []
top1andkl100s_min = []
top1andkl200s_min = []
top1andkl500s_min = []
justs_random = []
justs_random_from_top1_category = []
justs_random_from_top2_category = []
justs_random_from_top4_category = []
justs_random_from_top8_category = []
justs_random_from_top16_category = []
justs_random_from_top40_category = []
justs_random_from_top100_category = []
justs_random_from_top1_entropy = []
justs_random_from_top2_entropy = []
justs_random_from_top4_entropy = []
justs_random_from_top8_entropy = []
justs_random_from_top16_entropy = []
justs_random_from_top40_entropy = []
justs_random_from_top100_entropy = []
justs_random_from_top1_entropy_min = []
justs_random_from_top2_entropy_min = []
justs_random_from_top4_entropy_min = []
justs_random_from_top8_entropy_min = []
justs_random_from_top16_entropy_min = []
justs_random_from_top40_entropy_min = []
justs_random_from_top100_entropy_min = []
justs_random_from_top1_stderr = []
justs_random_from_top2_stderr = []
justs_random_from_top4_stderr = []
justs_random_from_top8_stderr = []
justs_random_from_top16_stderr = []
justs_random_from_top40_stderr = []
justs_random_from_top100_stderr = []
justs_random_from_top1_stderr_min = []
justs_random_from_top2_stderr_min = []
justs_random_from_top4_stderr_min = []
justs_random_from_top8_stderr_min = []
justs_random_from_top16_stderr_min = []
justs_random_from_top40_stderr_min = []
justs_random_from_top100_stderr_min = []
justs_random_from_top1_kl = []
justs_random_from_top2_kl = []
justs_random_from_top4_kl = []
justs_random_from_top8_kl = []
justs_random_from_top16_kl = []
justs_random_from_top40_kl = []
justs_random_from_top100_kl = []
justs_random_from_top1_kl_min = []
justs_random_from_top2_kl_min = []
justs_random_from_top4_kl_min = []
justs_random_from_top8_kl_min = []
justs_random_from_top16_kl_min = []
justs_random_from_top40_kl_min = []
justs_random_from_top100_kl_min = []

time_recordings = {}

scan_ids = os.listdir(prediction_root)
for i_s, scan_id in enumerate(scan_ids):
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
                    # bottom = sorted_idx[1:]
                    # topn = bottom[:n]
                    rank = np.argsort(pred_probs_mean, axis=1)[:, ::-1]
                    gt_rank = np.where(rank == gt_ind)[1]
                    # first try all with rank < n
                    # topn = np.where((0 < gt_rank) * (gt_rank < n))[0]
                    topn = np.where(gt_rank < n)[0]
                    topn = topn[topn != top_1]
                    if len(topn) == 0:
                        topn = np.argsort(gt_rank)[:2]
                        topn = topn[topn != top_1]

                    # entropies = [pred_instances[i]["mean_cls_entropy"] for i in topn]
                    # manually calculate entropy
                    entropies = [-np.sum(pred_probs_mean[i] * np.log(pred_probs_mean[i] + 1e-6)) for i in topn]
                    # sort by entropy 
                    sorted_idx = np.argsort(entropies)[::-1]
                    top_1_ent = topn[sorted_idx[0]] if not min_ent else topn[sorted_idx[-1]]
                    top_k_instances = [pred_instances[top_1]] + [pred_instances[top_1_ent]]
                else:
                    rank = np.argsort(pred_probs_mean, axis=1)[:, ::-1]
                    gt_rank = np.where(rank == gt_ind)[1]
                    # first try all with rank < n
                    # topn = np.where((0 < gt_rank) * (gt_rank < n))[0]
                    topn = np.where(gt_rank < n)[0]
                    if len(topn) == 0:
                        topn = np.argsort(gt_rank)[:1]
                    # print(len(topn))
                    entropies = [pred_instances[i]["mean_cls_entropy"] for i in topn]
                    # sort by entropy
                    sorted_idx = np.argsort(entropies)[::-1]
                    top_1_ent = (topn[sorted_idx[0]] if not min_ent else topn[sorted_idx[-1]]) 
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
                    top_k_instances = [pred_instances[top_1]] + [pred_instances[np.random.choice(len(pred_instances))]]
                else:
                    top_k_instances = [pred_instances[np.random.choice(len(pred_instances))]]
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
                    top_1 = sorted_idx[0]
                    rank = np.argsort(pred_probs_mean, axis=1)[:, ::-1]
                    gt_rank = np.where(rank == gt_ind)[1]
                    # first try all with rank < n
                    # topn = np.where((0 < gt_rank) * (gt_rank < n))[0]
                    topn = np.where(gt_rank < n)[0]
                    if len(topn) == 0:
                        topn = np.argsort(gt_rank)[:2]
                        topn = topn[topn != top_1]
                    topn = topn[topn != top_1]
                    top_k_instances = [pred_instances[top_1]] + [pred_instances[i] for i in topn]
                else:
                    rank = np.argsort(pred_probs_mean, axis=1)[:, ::-1]
                    gt_rank = np.where(rank == gt_ind)[1]
                    # first try all with rank < n
                    # topn = np.where((0 < gt_rank) * (gt_rank < n))[0]
                    topn = np.where(gt_rank < n)[0]
                    if len(topn) == 0:
                        topn = np.argsort(gt_rank)[:1]
                    top_k_instances = [pred_instances[i] for i in topn]

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
                # bottom = sorted_idx[1:]
                # topn = bottom[:n]
                rank = np.argsort(pred_probs_mean, axis=1)[:, ::-1]
                gt_rank = np.where(rank == gt_ind)[1]
                # first try all with rank < n
                topn = np.where((0 < gt_rank) * (gt_rank < n))[0]
                topn = topn[topn != top_1]
                if len(topn) == 0:
                    topn = np.argsort(gt_rank)[:2]
                    topn = topn[topn != top_1]
                # exclude the top_1
                topn = topn[topn != top_1]
                
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
                    # bottom = sorted_idx[1:]
                    # topn = bottom[:n]
                    rank = np.argsort(pred_probs_mean, axis=1)[:, ::-1]
                    gt_rank = np.where(rank == gt_ind)[1]
                    # first try all with rank < n
                    # topn = np.where((0 < gt_rank) * (gt_rank < n))[0]
                    topn = np.where(gt_rank < n)[0]
                    topn = topn[topn != top_1]
                    if len(topn) == 0:
                        topn = np.argsort(gt_rank)[:2]
                        topn = topn[topn != top_1]
                    # exclude the top_1
                    topn = topn[topn != top_1]

                    stds = []
                    for i in topn:
                        # manually calculate stderr
                        non_zero_views = np.where(pred_nonzero_views[i])[0]
                        non_zero_features = pred_mask2feats[i][non_zero_views]
                        pred_variace = np.var(non_zero_features, axis=0)
                        total_std = np.sqrt(np.sum(pred_variace)) if len(non_zero_views) > 1 else float("inf")
                        total_stderr = total_std / np.sqrt(len(non_zero_views))
                        stds.append(total_stderr)

                    # sort by entropy 
                    sorted_idx = np.argsort(stds)[::-1]
                    top_1_std = topn[sorted_idx[0]] if not min_ent else topn[sorted_idx[-1]]
                    top_k_instances = [pred_instances[top_1]] + [pred_instances[top_1_std]]
                else:
                    rank = np.argsort(pred_probs_mean, axis=1)[:, ::-1]
                    gt_rank = np.where(rank == gt_ind)[1]
                    # first try all with rank < n
                    # topn = np.where((0 < gt_rank) * (gt_rank < n))[0]
                    topn = np.where(gt_rank < n)[0]
                    if len(topn) == 0:
                        topn = np.argsort(gt_rank)[:1]
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

                    # bottom = sorted_idx[1:]
                    # topn = bottom[:n]
                    # calc the rank of the gt class for each pred
                    rank = np.argsort(pred_probs_mean, axis=1)[:, ::-1]
                    gt_rank = np.where(rank == gt_ind)[1]
                    # first try all with rank < n
                    # topn = np.where((0 < gt_rank) * (gt_rank < n))[0]
                    topn = np.where(gt_rank < n)[0]
                    topn = topn[topn != top_1]
                    if len(topn) == 0:
                        topn = np.argsort(gt_rank)[:2]
                        topn = topn[topn != top_1]
                    # exclude the top_1
                    topn = topn[topn != top_1]

                    kls = [mean_pairwise_kl(pred_probs[i]) for i in topn]
                    # sort by entropy 
                    sorted_idx = np.argsort(kls)[::-1]
                    top_1_kl = topn[sorted_idx[0]] if not min_ent else topn[sorted_idx[-1]]
                    top_k_instances = [pred_instances[top_1]] + [pred_instances[top_1_kl]]
                else:
                    rank = np.argsort(pred_probs_mean, axis=1)[:, ::-1]
                    gt_rank = np.where(rank == gt_ind)[1]
                    # first try all with rank < n
                    # topn = np.where((0 < gt_rank) * (gt_rank < n))[0]
                    topn = np.where(gt_rank < n)[0]
                    if len(topn) == 0:
                        topn = np.argsort(gt_rank)[:1]
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
        
        def mean_top_1_acc_top_n_hybrid(n, iou_threshold=0.25, min_ent=False, kl_ratio=1.0):
            results = []
            for gt_instance in gt_instances:
                # raw_feat = gt_instance["raw_feat"]
                # scores = (pred_features @ raw_feat.T)[:, :, 0]
                gt_ind = gt_instance["raw_label_idx"]
                mean_scores = pred_probs_mean[:, gt_ind]

                # mean_scores = np.mean(scores, axis=1)
                sorted_idx = np.argsort(mean_scores)[::-1]
                top_1 = sorted_idx[0]

                # bottom = sorted_idx[1:]
                # topn = bottom[:n]
                # calc the rank of the gt class for each pred
                rank = np.argsort(pred_probs_mean, axis=1)[:, ::-1]
                gt_rank = np.where(rank == gt_ind)[1]
                # first try all with rank < n
                topn = np.where((0 < gt_rank) * (gt_rank < n))[0]
                topn = topn[topn != top_1]
                if len(topn) == 0:
                    topn = np.argsort(gt_rank)[:2]
                    topn = topn[topn != top_1]
                # exclude the top_1
                topn = topn[topn != top_1]

                entropies = [pred_instances[i]["mean_cls_entropy"] for i in topn]
                kls = [mean_pairwise_kl(pred_probs[i]) for i in topn]
                rank_ent = np.argsort(entropies)[::-1].argsort()
                rank_kl = np.argsort(kls).argsort()
                combined_rank = rank_ent + kl_ratio * rank_kl
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
    
        
        t0 = time.time()
        top1 = mean_top_k_acc(1)
        time_recordings["top1s"] = time_recordings.get("top1s", []) + [time.time() - t0]
        t0 = time.time()
        top1_random = mean_top_1_and_random()
        time_recordings["top1_randoms"] = time_recordings.get("top1_randoms", []) + [time.time() - t0]
        t0 = time.time()
        top1_random_top_2 = mean_top_1_and_random_from_top_n(2)
        time_recordings["top1_random_top_2s"] = time_recordings.get("top1_random_top_2s", []) + [time.time() - t0]
        t0 = time.time()
        top1_random_top_4 = mean_top_1_and_random_from_top_n(4)
        time_recordings["top1_random_top_4s"] = time_recordings.get("top1_random_top_4s", []) + [time.time() - t0]
        t0 = time.time()
        top1_random_top_8 = mean_top_1_and_random_from_top_n(8)
        time_recordings["top1_random_top_8s"] = time_recordings.get("top1_random_top_8s", []) + [time.time() - t0]
        t0 = time.time()
        top1_random_top_16 = mean_top_1_and_random_from_top_n(16)
        time_recordings["top1_random_top_16s"] = time_recordings.get("top1_random_top_16s", []) + [time.time() - t0]
        t0 = time.time()
        top1_random_top_40 = mean_top_1_and_random_from_top_n(40)
        time_recordings["top1_random_top_40s"] = time_recordings.get("top1_random_top_40s", []) + [time.time() - t0]
        t0 = time.time()
        top1_random_top_100 = mean_top_1_and_random_from_top_n(100)
        time_recordings["top1_random_top_100s"] = time_recordings.get("top1_random_top_100s", []) + [time.time() - t0]

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
        top1andentropy_100= mean_top_1_acc_top_n_entropy(100)
        time_recordings["top1andentropy_100s"] = time_recordings.get("top1andentropy_100s", []) + [time.time() - t0]

        time_recordings["total_gt_entries"] = time_recordings.get("total_gt_entries", []) + [len(gt_instances)]
        # top1andentropy_200= mean_top_1_acc_top_n_entropy(1)
        # top1andentropy_500= mean_top_1_acc_top_n_entropy(1)

        top1andentropy_1_min = mean_top_1_acc_top_n_entropy(1, min_ent=True)
        top1andentropy_2_min = mean_top_1_acc_top_n_entropy(2, min_ent=True)
        top1andentropy_4_min = mean_top_1_acc_top_n_entropy(4, min_ent=True)
        top1andentropy_8_min = mean_top_1_acc_top_n_entropy(8, min_ent=True)
        top1andentropy_16_min = mean_top_1_acc_top_n_entropy(16, min_ent=True)
        top1andentropy_40_min = mean_top_1_acc_top_n_entropy(40, min_ent=True)
        top1andentropy_100_min = mean_top_1_acc_top_n_entropy(100, min_ent=True)
        # top1andentropy_200_min = mean_top_1_acc_top_n_entropy(1, min_ent=True)
        # top1andentropy_500_min = mean_top_1_acc_top_n_entropy(1, min_ent=True)

        top1andstd_1 = mean_top_1_acc_top_n_std(1)
        top1andstd_2 = mean_top_1_acc_top_n_std(2)
        top1andstd_4 = mean_top_1_acc_top_n_std(4)
        top1andstd_8 = mean_top_1_acc_top_n_std(8)
        top1andstd_16 = mean_top_1_acc_top_n_std(16)
        top1andstd_40 = mean_top_1_acc_top_n_std(40)
        top1andstd_100 = mean_top_1_acc_top_n_std(100)
        # top1andstd_200 = mean_top_1_acc_top_n_std(1)
        # top1andstd_500 = mean_top_1_acc_top_n_std(1)

        top1andstd_1_min = mean_top_1_acc_top_n_std(1, min_ent=True)
        top1andstd_2_min = mean_top_1_acc_top_n_std(2, min_ent=True)
        top1andstd_4_min = mean_top_1_acc_top_n_std(4, min_ent=True)
        top1andstd_8_min = mean_top_1_acc_top_n_std(8, min_ent=True)
        top1andstd_16_min = mean_top_1_acc_top_n_std(16, min_ent=True)
        top1andstd_40_min = mean_top_1_acc_top_n_std(40, min_ent=True)
        top1andstd_100_min = mean_top_1_acc_top_n_std(100, min_ent=True)
        # top1andstd_200_min = mean_top_1_acc_top_n_std(1, min_ent=True)
        # top1andstd_500_min = mean_top_1_acc_top_n_std(1, min_ent=True)

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
        top1andstderr_100 = mean_top_1_acc_top_n_stderr(100)
        time_recordings["top1andstderr_100s"] = time_recordings.get("top1andstderr_100s", []) + [time.time() - t0]
        # top1andstderr_200 = mean_top_1_acc_top_n_stderr(1)
        # top1andstderr_500 = mean_top_1_acc_top_n_stderr(1)

        top1andstderr_1_min = mean_top_1_acc_top_n_stderr(1, min_ent=True)
        top1andstderr_2_min = mean_top_1_acc_top_n_stderr(2, min_ent=True)
        top1andstderr_4_min = mean_top_1_acc_top_n_stderr(4, min_ent=True)
        top1andstderr_8_min = mean_top_1_acc_top_n_stderr(8, min_ent=True)
        top1andstderr_16_min = mean_top_1_acc_top_n_stderr(16, min_ent=True)
        top1andstderr_40_min = mean_top_1_acc_top_n_stderr(40, min_ent=True)
        top1andstderr_100_min = mean_top_1_acc_top_n_stderr(100, min_ent=True)
        # top1andstderr_200_min = mean_top_1_acc_top_n_stderr(1, min_ent=True)
        # top1andstderr_500_min = mean_top_1_acc_top_n_stderr(1, min_ent=True)

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
        top1andkl_100 = mean_top_1_acc_top_n_kl(100)
        time_recordings["top1andkl_100s"] = time_recordings.get("top1andkl_100s", []) + [time.time() - t0]
        # top1andkl_200 = mean_top_1_acc_top_n_kl(1)
        # top1andkl_500 = mean_top_1_acc_top_n_kl(1)

        top1andkl_1_min = mean_top_1_acc_top_n_kl(1, min_ent=True)
        top1andkl_2_min = mean_top_1_acc_top_n_kl(2, min_ent=True)
        top1andkl_4_min = mean_top_1_acc_top_n_kl(4, min_ent=True)
        top1andkl_8_min = mean_top_1_acc_top_n_kl(8, min_ent=True)
        top1andkl_16_min = mean_top_1_acc_top_n_kl(16, min_ent=True)
        top1andkl_40_min = mean_top_1_acc_top_n_kl(40, min_ent=True)
        top1andkl_100_min = mean_top_1_acc_top_n_kl(100, min_ent=True)
        # top1andkl_200_min = mean_top_1_acc_top_n_kl(200, min_ent=True)
        # top1andkl_500_min = mean_top_1_acc_top_n_kl(1, min_ent=True)

        just_random = mean_top_1_and_random(skip_top1=False)
        
        just_random_from_top1_category = mean_top_1_and_random_from_top_n(1, skip_top1=True)
        just_random_from_top2_category = mean_top_1_and_random_from_top_n(2, skip_top1=True)
        just_random_from_top4_category = mean_top_1_and_random_from_top_n(4, skip_top1=True)
        just_random_from_top8_category = mean_top_1_and_random_from_top_n(8, skip_top1=True)
        just_random_from_top16_category = mean_top_1_and_random_from_top_n(16, skip_top1=True)
        just_random_from_top40_category = mean_top_1_and_random_from_top_n(40, skip_top1=True)
        just_random_from_top100_category = mean_top_1_and_random_from_top_n(100, skip_top1=True)

        just_random_from_top1_entropy = mean_top_1_acc_top_n_entropy(1, skip_top1=True)
        just_random_from_top2_entropy = mean_top_1_acc_top_n_entropy(2, skip_top1=True)
        just_random_from_top4_entropy = mean_top_1_acc_top_n_entropy(4, skip_top1=True)
        just_random_from_top8_entropy = mean_top_1_acc_top_n_entropy(8, skip_top1=True)
        just_random_from_top16_entropy = mean_top_1_acc_top_n_entropy(16, skip_top1=True)
        just_random_from_top40_entropy = mean_top_1_acc_top_n_entropy(40, skip_top1=True)
        just_random_from_top100_entropy = mean_top_1_acc_top_n_entropy(100, skip_top1=True)

        just_random_from_top1_entropy_min = mean_top_1_acc_top_n_entropy(1, skip_top1=True, min_ent=True)
        just_random_from_top2_entropy_min = mean_top_1_acc_top_n_entropy(2, skip_top1=True, min_ent=True)
        just_random_from_top4_entropy_min = mean_top_1_acc_top_n_entropy(4, skip_top1=True, min_ent=True)
        just_random_from_top8_entropy_min = mean_top_1_acc_top_n_entropy(8, skip_top1=True, min_ent=True)
        just_random_from_top16_entropy_min = mean_top_1_acc_top_n_entropy(16, skip_top1=True, min_ent=True)
        just_random_from_top40_entropy_min = mean_top_1_acc_top_n_entropy(40, skip_top1=True, min_ent=True)
        just_random_from_top100_entropy_min = mean_top_1_acc_top_n_entropy(100, skip_top1=True, min_ent=True)

        just_random_from_top1_stderr = mean_top_1_acc_top_n_stderr(1, skip_top1=True)
        just_random_from_top2_stderr = mean_top_1_acc_top_n_stderr(2, skip_top1=True)
        just_random_from_top4_stderr = mean_top_1_acc_top_n_stderr(4, skip_top1=True)
        just_random_from_top8_stderr = mean_top_1_acc_top_n_stderr(8, skip_top1=True)
        just_random_from_top16_stderr = mean_top_1_acc_top_n_stderr(16, skip_top1=True)
        just_random_from_top40_stderr = mean_top_1_acc_top_n_stderr(40, skip_top1=True)
        just_random_from_top100_stderr = mean_top_1_acc_top_n_stderr(100, skip_top1=True)

        just_random_from_top1_stderr_min = mean_top_1_acc_top_n_stderr(1, skip_top1=True, min_ent=True)
        just_random_from_top2_stderr_min = mean_top_1_acc_top_n_stderr(2, skip_top1=True, min_ent=True)
        just_random_from_top4_stderr_min = mean_top_1_acc_top_n_stderr(4, skip_top1=True, min_ent=True)
        just_random_from_top8_stderr_min = mean_top_1_acc_top_n_stderr(8, skip_top1=True, min_ent=True)
        just_random_from_top16_stderr_min = mean_top_1_acc_top_n_stderr(16, skip_top1=True, min_ent=True)
        just_random_from_top40_stderr_min = mean_top_1_acc_top_n_stderr(40, skip_top1=True, min_ent=True)
        just_random_from_top100_stderr_min = mean_top_1_acc_top_n_stderr(100, skip_top1=True, min_ent=True)

        just_random_from_top1_kl = mean_top_1_acc_top_n_kl(1, skip_top1=True)
        just_random_from_top2_kl = mean_top_1_acc_top_n_kl(2, skip_top1=True)
        just_random_from_top4_kl = mean_top_1_acc_top_n_kl(4, skip_top1=True)
        just_random_from_top8_kl = mean_top_1_acc_top_n_kl(8, skip_top1=True)
        just_random_from_top16_kl = mean_top_1_acc_top_n_kl(16, skip_top1=True)
        just_random_from_top40_kl = mean_top_1_acc_top_n_kl(40, skip_top1=True)
        just_random_from_top100_kl = mean_top_1_acc_top_n_kl(100, skip_top1=True)

        just_random_from_top1_kl_min = mean_top_1_acc_top_n_kl(1, skip_top1=True, min_ent=True)
        just_random_from_top2_kl_min = mean_top_1_acc_top_n_kl(2, skip_top1=True, min_ent=True)
        just_random_from_top4_kl_min = mean_top_1_acc_top_n_kl(4, skip_top1=True, min_ent=True)
        just_random_from_top8_kl_min = mean_top_1_acc_top_n_kl(8, skip_top1=True, min_ent=True)
        just_random_from_top16_kl_min = mean_top_1_acc_top_n_kl(16, skip_top1=True, min_ent=True)
        just_random_from_top40_kl_min = mean_top_1_acc_top_n_kl(40, skip_top1=True, min_ent=True)
        just_random_from_top100_kl_min = mean_top_1_acc_top_n_kl(100, skip_top1=True, min_ent=True)

        top1s.append(top1)
        top2s.append(top2)
        top1s_random.append(top1_random)
        top1s_random_top_2.append(top1_random_top_2)
        top1s_random_top_4.append(top1_random_top_4)
        top1s_random_top_8.append(top1_random_top_8)
        top1s_random_top_16.append(top1_random_top_16)
        top1s_random_top_40.append(top1_random_top_40)
        top1s_random_top_100.append(top1_random_top_100)
        top1ent1s.append(top1andentropy_1)
        top1ent2s.append(top1andentropy_2)
        top1ent4s.append(top1andentropy_4)
        top1ent8s.append(top1andentropy_8)
        top1ent16s.append(top1andentropy_16)
        top1ent40s.append(top1andentropy_40)
        top1ent100s.append(top1andentropy_100)  
        # top1ent200s.append(top1andentropy_200)
        # top1ent500s.append(top1andentropy_500)
        top1ent1s_min.append(top1andentropy_1_min)
        top1ent2s_min.append(top1andentropy_2_min)
        top1ent4s_min.append(top1andentropy_4_min)
        top1ent8s_min.append(top1andentropy_8_min)
        top1ent16s_min.append(top1andentropy_16_min)
        top1ent40s_min.append(top1andentropy_40_min)
        top1ent100s_min.append(top1andentropy_100_min)
        # top1ent200s_min.append(top1andentropy_200_min)
        # top1ent500s_min.append(top1andentropy_500_min)
        top1std1s.append(top1andstd_1)
        top1std2s.append(top1andstd_2)
        top1std4s.append(top1andstd_4)
        top1std8s.append(top1andstd_8)
        top1std16s.append(top1andstd_16)
        top1std40s.append(top1andstd_40)
        top1std100s.append(top1andstd_100)
        # top1std200s.append(top1andstd_200)
        # top1std500s.append(top1andstd_500)
        top1std1s_min.append(top1andstd_1_min)
        top1std2s_min.append(top1andstd_2_min)
        top1std4s_min.append(top1andstd_4_min)
        top1std8s_min.append(top1andstd_8_min)
        top1std16s_min.append(top1andstd_16_min)
        top1std40s_min.append(top1andstd_40_min)
        top1std100s_min.append(top1andstd_100_min)
        # top1std200s_min.append(top1andstd_200_min)
        # top1std500s_min.append(top1andstd_500_min)
        top1stderr1s.append(top1andstderr_1)
        top1stderr2s.append(top1andstderr_2)
        top1stderr4s.append(top1andstderr_4)
        top1stderr8s.append(top1andstderr_8)
        top1stderr16s.append(top1andstderr_16)
        top1stderr40s.append(top1andstderr_40)
        top1stderr100s.append(top1andstderr_100)
        # top1stderr200s.append(top1andstderr_200)
        # top1stderr500s.append(top1andstderr_500)
        top1stderr1s_min.append(top1andstderr_1_min)
        top1stderr2s_min.append(top1andstderr_2_min)
        top1stderr4s_min.append(top1andstderr_4_min)
        top1stderr8s_min.append(top1andstderr_8_min)
        top1stderr16s_min.append(top1andstderr_16_min)
        top1stderr40s_min.append(top1andstderr_40_min)
        top1stderr100s_min.append(top1andstderr_100_min)
        # top1stderr200s_min.append(top1andstderr_200_min)
        # top1stderr500s_min.append(top1andstderr_500_min)
        top1andkl1s.append(top1andkl_1)
        top1andkl2s.append(top1andkl_2)
        top1andkl4s.append(top1andkl_4)
        top1andkl8s.append(top1andkl_8)
        top1andkl16s.append(top1andkl_16)
        top1andkl40s.append(top1andkl_40)
        top1andkl100s.append(top1andkl_100)
        # top1andkl200s.append(top1andkl_200)
        # top1andkl500s.append(top1andkl_500)
        top1andkl1s_min.append(top1andkl_1_min)
        top1andkl2s_min.append(top1andkl_2_min)
        top1andkl4s_min.append(top1andkl_4_min)
        top1andkl8s_min.append(top1andkl_8_min)
        top1andkl16s_min.append(top1andkl_16_min)
        top1andkl40s_min.append(top1andkl_40_min)
        top1andkl100s_min.append(top1andkl_100_min)
        # top1andkl200s_min.append(top1andkl_200_min)
        # top1andkl500s_min.append(top1andkl_500_min)
        justs_random.append(just_random)
        justs_random_from_top1_category.append(just_random_from_top1_category)
        justs_random_from_top2_category.append(just_random_from_top2_category)
        justs_random_from_top4_category.append(just_random_from_top4_category)  
        justs_random_from_top8_category.append(just_random_from_top8_category)
        justs_random_from_top16_category.append(just_random_from_top16_category)
        justs_random_from_top40_category.append(just_random_from_top40_category)
        justs_random_from_top100_category.append(just_random_from_top100_category)
        justs_random_from_top1_entropy.append(just_random_from_top1_entropy)
        justs_random_from_top2_entropy.append(just_random_from_top2_entropy)
        justs_random_from_top4_entropy.append(just_random_from_top4_entropy)
        justs_random_from_top8_entropy.append(just_random_from_top8_entropy)
        justs_random_from_top16_entropy.append(just_random_from_top16_entropy)
        justs_random_from_top40_entropy.append(just_random_from_top40_entropy)
        justs_random_from_top100_entropy.append(just_random_from_top100_entropy)
        justs_random_from_top1_entropy_min.append(just_random_from_top1_entropy_min)
        justs_random_from_top2_entropy_min.append(just_random_from_top2_entropy_min)
        justs_random_from_top4_entropy_min.append(just_random_from_top4_entropy_min)
        justs_random_from_top8_entropy_min.append(just_random_from_top8_entropy_min)
        justs_random_from_top16_entropy_min.append(just_random_from_top16_entropy_min)
        justs_random_from_top40_entropy_min.append(just_random_from_top40_entropy_min)
        justs_random_from_top100_entropy_min.append(just_random_from_top100_entropy_min)
        justs_random_from_top1_stderr.append(just_random_from_top1_stderr)
        justs_random_from_top2_stderr.append(just_random_from_top2_stderr)
        justs_random_from_top4_stderr.append(just_random_from_top4_stderr)
        justs_random_from_top8_stderr.append(just_random_from_top8_stderr)
        justs_random_from_top16_stderr.append(just_random_from_top16_stderr)
        justs_random_from_top40_stderr.append(just_random_from_top40_stderr)
        justs_random_from_top100_stderr.append(just_random_from_top100_stderr)
        justs_random_from_top1_stderr_min.append(just_random_from_top1_stderr_min)
        justs_random_from_top2_stderr_min.append(just_random_from_top2_stderr_min)
        justs_random_from_top4_stderr_min.append(just_random_from_top4_stderr_min)
        justs_random_from_top8_stderr_min.append(just_random_from_top8_stderr_min)
        justs_random_from_top16_stderr_min.append(just_random_from_top16_stderr_min)
        justs_random_from_top40_stderr_min.append(just_random_from_top40_stderr_min)
        justs_random_from_top100_stderr_min.append(just_random_from_top100_stderr_min)
        justs_random_from_top1_kl.append(just_random_from_top1_kl)
        justs_random_from_top2_kl.append(just_random_from_top2_kl)
        justs_random_from_top4_kl.append(just_random_from_top4_kl)
        justs_random_from_top8_kl.append(just_random_from_top8_kl)
        justs_random_from_top16_kl.append(just_random_from_top16_kl)
        justs_random_from_top40_kl.append(just_random_from_top40_kl)
        justs_random_from_top100_kl.append(just_random_from_top100_kl)
        justs_random_from_top1_kl_min.append(just_random_from_top1_kl_min)
        justs_random_from_top2_kl_min.append(just_random_from_top2_kl_min)
        justs_random_from_top4_kl_min.append(just_random_from_top4_kl_min)
        justs_random_from_top8_kl_min.append(just_random_from_top8_kl_min)
        justs_random_from_top16_kl_min.append(just_random_from_top16_kl_min)
        justs_random_from_top40_kl_min.append(just_random_from_top40_kl_min)
        justs_random_from_top100_kl_min.append(just_random_from_top100_kl_min)

### save the raw data
import json
with open(f"time_raw_category.json", "w") as f:
    json.dump(time_recordings, f, indent=4)

top1 = np.mean(top1s)
top2 = np.mean(top2s)
top1_random = np.mean(top1s_random)
top1_random_top_2 = np.mean(top1s_random_top_2)
top1_random_top_4 = np.mean(top1s_random_top_4)
top1_random_top_8 = np.mean(top1s_random_top_8)
top1_random_top_16 = np.mean(top1s_random_top_16)
top1_random_top_40 = np.mean(top1s_random_top_40)
top1_random_top_100 = np.mean(top1s_random_top_100)
top1_entropy = np.mean(top1ent1s)
top2_entropy = np.mean(top1ent2s)
top4_entropy = np.mean(top1ent4s)
top8_entropy = np.mean(top1ent8s)
top16_entropy = np.mean(top1ent16s)
top40_entropy = np.mean(top1ent40s)
top100_entropy = np.mean(top1ent100s)
# top200_entropy = np.mean(top1ent200s)
# top500_entropy = np.mean(top1ent500s)
top1_entropy_min = np.mean(top1ent1s_min)
top2_entropy_min = np.mean(top1ent2s_min)
top4_entropy_min = np.mean(top1ent4s_min)
top8_entropy_min = np.mean(top1ent8s_min)
top16_entropy_min = np.mean(top1ent16s_min)
top40_entropy_min = np.mean(top1ent40s_min)
top100_entropy_min = np.mean(top1ent100s_min)
# top200_entropy_min = np.mean(top1ent200s_min)
# top500_entropy_min = np.mean(top1ent500s_min)
top1_std = np.mean(top1std1s)
top2_std = np.mean(top1std2s)
top4_std = np.mean(top1std4s)
top8_std = np.mean(top1std8s)
top16_std = np.mean(top1std16s)
top40_std = np.mean(top1std40s)
top100_std = np.mean(top1std100s)
# top200_std = np.mean(top1std200s)
# top500_std = np.mean(top1std500s)
top1_std_min = np.mean(top1std1s_min)
top2_std_min = np.mean(top1std2s_min)
top4_std_min = np.mean(top1std4s_min)
top8_std_min = np.mean(top1std8s_min)
top16_std_min = np.mean(top1std16s_min)
top40_std_min = np.mean(top1std40s_min)
top100_std_min = np.mean(top1std100s_min)
# top200_std_min = np.mean(top1std200s_min)
# top500_std_min = np.mean(top1std500s_min)
top1_stderr = np.mean(top1stderr1s)
top2_stderr = np.mean(top1stderr2s)
top4_stderr = np.mean(top1stderr4s)
top8_stderr = np.mean(top1stderr8s)
top16_stderr = np.mean(top1stderr16s)
top40_stderr = np.mean(top1stderr40s)
top100_stderr = np.mean(top1stderr100s)
# top200_stderr = np.mean(top1stderr200s)
# top500_stderr = np.mean(top1stderr500s)
top1_stderr_min = np.mean(top1stderr1s_min)
top2_stderr_min = np.mean(top1stderr2s_min)
top4_stderr_min = np.mean(top1stderr4s_min)
top8_stderr_min = np.mean(top1stderr8s_min)
top16_stderr_min = np.mean(top1stderr16s_min)
top40_stderr_min = np.mean(top1stderr40s_min)
top100_stderr_min = np.mean(top1stderr100s_min)
# top200_stderr_min = np.mean(top1stderr200s_min)
# top500_stderr_min = np.mean(top1stderr500s_min)
top1andkl1 = np.mean(top1andkl1s)
top1andkl2 = np.mean(top1andkl2s)
top1andkl4 = np.mean(top1andkl4s)
top1andkl8 = np.mean(top1andkl8s)
top1andkl16 = np.mean(top1andkl16s)
top1andkl40 = np.mean(top1andkl40s)
top1andkl100 = np.mean(top1andkl100s)
# top1andkl200 = np.mean(top1andkl200s)
# top1andkl500 = np.mean(top1andkl500s)
top1andkl1_min = np.mean(top1andkl1s_min)
top1andkl2_min = np.mean(top1andkl2s_min)
top1andkl4_min = np.mean(top1andkl4s_min)
top1andkl8_min = np.mean(top1andkl8s_min)
top1andkl16_min = np.mean(top1andkl16s_min)
top1andkl40_min = np.mean(top1andkl40s_min)
top1andkl100_min = np.mean(top1andkl100s_min)
# top1andkl200_min = np.mean(top1andkl200s_min)
# top1andkl500_min = np.mean(top1andkl500s_min)
just_random = np.mean(justs_random)
just_random_from_top1_category = np.mean(justs_random_from_top1_category)
just_random_from_top2_category = np.mean(justs_random_from_top2_category)
just_random_from_top4_category = np.mean(justs_random_from_top4_category)
just_random_from_top8_category = np.mean(justs_random_from_top8_category)
just_random_from_top16_category = np.mean(justs_random_from_top16_category)
just_random_from_top40_category = np.mean(justs_random_from_top40_category)
just_random_from_top100_category = np.mean(justs_random_from_top100_category)
just_random_from_top1_entropy = np.mean(justs_random_from_top1_entropy)
just_random_from_top2_entropy = np.mean(justs_random_from_top2_entropy)
just_random_from_top4_entropy = np.mean(justs_random_from_top4_entropy)
just_random_from_top8_entropy = np.mean(justs_random_from_top8_entropy)
just_random_from_top16_entropy = np.mean(justs_random_from_top16_entropy)
just_random_from_top40_entropy = np.mean(justs_random_from_top40_entropy)
just_random_from_top100_entropy = np.mean(justs_random_from_top100_entropy)
just_random_from_top1_entropy_min = np.mean(justs_random_from_top1_entropy_min)
just_random_from_top2_entropy_min = np.mean(justs_random_from_top2_entropy_min)
just_random_from_top4_entropy_min = np.mean(justs_random_from_top4_entropy_min)
just_random_from_top8_entropy_min = np.mean(justs_random_from_top8_entropy_min)
just_random_from_top16_entropy_min = np.mean(justs_random_from_top16_entropy_min)
just_random_from_top40_entropy_min = np.mean(justs_random_from_top40_entropy_min)
just_random_from_top100_entropy_min = np.mean(justs_random_from_top100_entropy_min)
just_random_from_top1_stderr = np.mean(justs_random_from_top1_stderr)
just_random_from_top2_stderr = np.mean(justs_random_from_top2_stderr)
just_random_from_top4_stderr = np.mean(justs_random_from_top4_stderr)
just_random_from_top8_stderr = np.mean(justs_random_from_top8_stderr)
just_random_from_top16_stderr = np.mean(justs_random_from_top16_stderr)
just_random_from_top40_stderr = np.mean(justs_random_from_top40_stderr)
just_random_from_top100_stderr = np.mean(justs_random_from_top100_stderr)
just_random_from_top1_stderr_min = np.mean(justs_random_from_top1_stderr_min)
just_random_from_top2_stderr_min = np.mean(justs_random_from_top2_stderr_min)
just_random_from_top4_stderr_min = np.mean(justs_random_from_top4_stderr_min)
just_random_from_top8_stderr_min = np.mean(justs_random_from_top8_stderr_min)
just_random_from_top16_stderr_min = np.mean(justs_random_from_top16_stderr_min)
just_random_from_top40_stderr_min = np.mean(justs_random_from_top40_stderr_min)
just_random_from_top100_stderr_min = np.mean(justs_random_from_top100_stderr_min)
just_random_from_top1_kl = np.mean(justs_random_from_top1_kl)
just_random_from_top2_kl = np.mean(justs_random_from_top2_kl)
just_random_from_top4_kl = np.mean(justs_random_from_top4_kl)
just_random_from_top8_kl = np.mean(justs_random_from_top8_kl)
just_random_from_top16_kl = np.mean(justs_random_from_top16_kl)
just_random_from_top40_kl = np.mean(justs_random_from_top40_kl)
just_random_from_top100_kl = np.mean(justs_random_from_top100_kl)
just_random_from_top1_kl_min = np.mean(justs_random_from_top1_kl_min)
just_random_from_top2_kl_min = np.mean(justs_random_from_top2_kl_min)
just_random_from_top4_kl_min = np.mean(justs_random_from_top4_kl_min)
just_random_from_top8_kl_min = np.mean(justs_random_from_top8_kl_min)
just_random_from_top16_kl_min = np.mean(justs_random_from_top16_kl_min)
just_random_from_top40_kl_min = np.mean(justs_random_from_top40_kl_min)
just_random_from_top100_kl_min = np.mean(justs_random_from_top100_kl_min)

output = {
    "top1": top1,
    "top1+random": top1_random,
    "top2": top2,
    "top1+random_from_topk_category": {
        "1": top2,
        "2": top1_random_top_2,
        "4": top1_random_top_4,
        "8": top1_random_top_8,
        "16": top1_random_top_16,
        "40": top1_random_top_40,
        "100": top1_random_top_100
    },
    "top1+min_entropy_from_topk_category": {
        "1": top1_entropy_min,
        "2": top2_entropy_min,
        "4": top4_entropy_min,
        "8": top8_entropy_min,
        "16": top16_entropy_min,
        "40": top40_entropy_min,
        "100": top100_entropy_min
    },
    "top1+max_entropy_from_topk_category": {
        "1": top1_entropy,
        "2": top2_entropy,
        "4": top4_entropy,
        "8": top8_entropy,
        "16": top16_entropy,
        "40": top40_entropy,
        "100": top100_entropy
    },
    "top1+min_kl_from_topk_category": {
        "1": top1andkl1_min,
        "2": top1andkl2_min,
        "4": top1andkl4_min,
        "8": top1andkl8_min,
        "16": top1andkl16_min,
        "40": top1andkl40_min,
        "100": top1andkl100_min
    },
    "top1+max_kl_from_topk_category": {
        "1": top1andkl1,
        "2": top1andkl2,
        "4": top1andkl4,
        "8": top1andkl8,
        "16": top1andkl16,
        "40": top1andkl40,
        "100": top1andkl100
    },
    "top1+min_stderr_from_topk_category": {
        "1": top1_stderr_min,
        "2": top2_stderr_min,
        "4": top4_stderr_min,
        "8": top8_stderr_min,
        "16": top16_stderr_min,
        "40": top40_stderr_min,
        "100": top100_stderr_min
    },
    "top1+max_stderr_from_topk_category": {
        "1": top1_stderr,
        "2": top2_stderr,
        "4": top4_stderr,
        "8": top8_stderr,
        "16": top16_stderr,
        "40": top40_stderr,
        "100": top100_stderr
    },
    "just_random": just_random,
    "just_random_from_topk_category": {
        "1": just_random_from_top1_category,
        "2": just_random_from_top2_category,
        "4": just_random_from_top4_category,
        "8": just_random_from_top8_category,
        "16": just_random_from_top16_category,
        "40": just_random_from_top40_category,
        "100": just_random_from_top100_category
    },
    "just_max_entropy_from_topk_category": {
        "1": just_random_from_top1_entropy,
        "2": just_random_from_top2_entropy,
        "4": just_random_from_top4_entropy,
        "8": just_random_from_top8_entropy,
        "16": just_random_from_top16_entropy,
        "40": just_random_from_top40_entropy,
        "100": just_random_from_top100_entropy
    },
    "just_min_entropy_from_topk_category": {
        "1": just_random_from_top1_entropy_min,
        "2": just_random_from_top2_entropy_min,
        "4": just_random_from_top4_entropy_min,
        "8": just_random_from_top8_entropy_min,
        "16": just_random_from_top16_entropy_min,
        "40": just_random_from_top40_entropy_min,
        "100": just_random_from_top100_entropy_min
    },
    "just_max_stderr_from_topk_category": {
        "1": just_random_from_top1_stderr,
        "2": just_random_from_top2_stderr,
        "4": just_random_from_top4_stderr,
        "8": just_random_from_top8_stderr,
        "16": just_random_from_top16_stderr,
        "40": just_random_from_top40_stderr,
        "100": just_random_from_top100_stderr
    },
    "just_min_stderr_from_topk_category": {
        "1": just_random_from_top1_stderr_min,
        "2": just_random_from_top2_stderr_min,
        "4": just_random_from_top4_stderr_min,
        "8": just_random_from_top8_stderr_min,
        "16": just_random_from_top16_stderr_min,
        "40": just_random_from_top40_stderr_min,
        "100": just_random_from_top100_stderr_min
    },
    "just_max_kl_from_topk_category": {
        "1": just_random_from_top1_kl,
        "2": just_random_from_top2_kl,
        "4": just_random_from_top4_kl,
        "8": just_random_from_top8_kl,
        "16": just_random_from_top16_kl,
        "40": just_random_from_top40_kl,
        "100": just_random_from_top100_kl
    },
    "just_min_kl_from_topk_category": {
        "1": just_random_from_top1_kl_min,
        "2": just_random_from_top2_kl_min,
        "4": just_random_from_top4_kl_min,
        "8": just_random_from_top8_kl_min,
        "16": just_random_from_top16_kl_min,
        "40": just_random_from_top40_kl_min,
        "100": just_random_from_top100_kl_min
    }

}

with open("topk_categoty_replanning.json", "w") as f:
    json.dump(output, f, indent=4)

# also record the raw data
output_raw = {
    "top1": top1s,
    "top1+random": top1s_random,
    "top2": top2s,
    "top1+random_from_topk_category": {
        "1": top2s,
        "2": top1s_random_top_2,
        "4": top1s_random_top_4,
        "8": top1s_random_top_8,
        "16": top1s_random_top_16,
        "40": top1s_random_top_40,
        "100": top1s_random_top_100
    },
    "top1+min_entropy_from_topk_category": {
        "1": top1ent1s_min,
        "2": top1ent2s_min,
        "4": top1ent4s_min,
        "8": top1ent8s_min,
        "16": top1ent16s_min,
        "40": top1ent40s_min,
        "100": top1ent100s_min
    },
    "top1+max_entropy_from_topk_category": {
        "1": top1ent1s,
        "2": top1ent2s,
        "4": top1ent4s,
        "8": top1ent8s,
        "16": top1ent16s,
        "40": top1ent40s,
        "100": top1ent100s
    },
    "top1+min_kl_from_topk_category": {
        "1": top1andkl1s_min,
        "2": top1andkl2s_min,
        "4": top1andkl4s_min,
        "8": top1andkl8s_min,
        "16": top1andkl16s_min,
        "40": top1andkl40s_min,
        "100": top1andkl100s_min
    },
    "top1+max_kl_from_topk_category": {
        "1": top1andkl1s,
        "2": top1andkl2s,
        "4": top1andkl4s,
        "8": top1andkl8s,
        "16": top1andkl16s,
        "40": top1andkl40s,
        "100": top1andkl100s
    },
    "top1+min_stderr_from_topk_category": {
        "1": top1stderr1s_min,
        "2": top1stderr2s_min,
        "4": top1stderr4s_min,
        "8": top1stderr8s_min,
        "16": top1stderr16s_min,
        "40": top1stderr40s_min,
        "100": top1stderr100s_min
    },
    "top1+max_stderr_from_topk_category": {
        "1": top1stderr1s,
        "2": top1stderr2s,
        "4": top1stderr4s,
        "8": top1stderr8s,
        "16": top1stderr16s,
        "40": top1stderr40s,
        "100": top1stderr100s
    },
    "just_random": justs_random,
    "just_random_from_topk_category": {
        "1": justs_random_from_top1_category,
        "2": justs_random_from_top2_category,
        "4": justs_random_from_top4_category,
        "8": justs_random_from_top8_category,
        "16": justs_random_from_top16_category,
        "40": justs_random_from_top40_category,
        "100": justs_random_from_top100_category
    },
    "just_max_entropy_from_topk_category": {
        "1": justs_random_from_top1_entropy,
        "2": justs_random_from_top2_entropy,
        "4": justs_random_from_top4_entropy,
        "8": justs_random_from_top8_entropy,
        "16": justs_random_from_top16_entropy,
        "40": justs_random_from_top40_entropy,
        "100": justs_random_from_top100_entropy
    },
    "just_min_entropy_from_topk_category": {
        "1": justs_random_from_top1_entropy_min,
        "2": justs_random_from_top2_entropy_min,
        "4": justs_random_from_top4_entropy_min,
        "8": justs_random_from_top8_entropy_min,
        "16": justs_random_from_top16_entropy_min,
        "40": justs_random_from_top40_entropy_min,
        "100": justs_random_from_top100_entropy_min
    },
    "just_max_stderr_from_topk_category": {
        "1": justs_random_from_top1_stderr,
        "2": justs_random_from_top2_stderr,
        "4": justs_random_from_top4_stderr,
        "8": justs_random_from_top8_stderr,
        "16": justs_random_from_top16_stderr,
        "40": justs_random_from_top40_stderr,
        "100": justs_random_from_top100_stderr
    },
    "just_min_stderr_from_topk_category": {
        "1": justs_random_from_top1_stderr_min,
        "2": justs_random_from_top2_stderr_min,
        "4": justs_random_from_top4_stderr_min,
        "8": justs_random_from_top8_stderr_min,
        "16": justs_random_from_top16_stderr_min,
        "40": justs_random_from_top40_stderr_min,
        "100": justs_random_from_top100_stderr_min
    },
    "just_max_kl_from_topk_category": {
        "1": justs_random_from_top1_kl,
        "2": justs_random_from_top2_kl,
        "4": justs_random_from_top4_kl,
        "8": justs_random_from_top8_kl,
        "16": justs_random_from_top16_kl,
        "40": justs_random_from_top40_kl,
        "100": justs_random_from_top100_kl
    },
    "just_min_kl_from_topk_category": {
        "1": justs_random_from_top1_kl_min,
        "2": justs_random_from_top2_kl_min,
        "4": justs_random_from_top4_kl_min,
        "8": justs_random_from_top8_kl_min,
        "16": justs_random_from_top16_kl_min,
        "40": justs_random_from_top40_kl_min,
        "100": justs_random_from_top100_kl_min
    }
}

with open("topk_categoty_replanning_raw.json", "w") as f:
    json.dump(output_raw, f, indent=4)

