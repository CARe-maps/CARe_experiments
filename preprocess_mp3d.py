mp3d_rgbd_dir = "./dataset/scans"
output_root='./resources/mp3d_output'

import os
from glob import glob
import numpy as np
import open3d
from tqdm import tqdm
from PIL import Image
import json


scan_ids = ['2azQ1b91cZZ',
 'TbHJrupSAjP',
 'zsNo4HB9uLZ',
 '8194nk5LbLH',
 'Z6MFQCViBuw',
 'EU6Fwq7SyZv',
 'X7HyMhZNoso',
 'x8F5xyUWy9e',
 'oLBMNvg9in8',
 'QUCTc6BB5sX'
]


#0. ply
#1. pose
#2. intrinsic (resolution), poses
#3. image links, depth link
#4. depth scale
def get_region_group(lookup_path="./node_region_lookups.json"):
    import json
    out_group = {}
    with open(lookup_path, "r") as f:
        lookup = json.load(f)
    scans = list(lookup.keys())
    for scan in scans:
        nodes = list(lookup[scan].keys())
        region_names = sorted(list(set(lookup[scan].values())), key=lambda x: int(x))
        region_nodes = {
            region: [node for node in nodes if lookup[scan][node] == region]
            for region in region_names
        }
        out_group[scan] = region_nodes
    return out_group

out_group = get_region_group()


print(glob(f"{mp3d_rgbd_dir}/*"))

for scan_dir in tqdm(glob(f"{mp3d_rgbd_dir}/*")):
    _, scan_id = os.path.split(scan_dir)
    if scan_id not in scan_ids:
        continue

    camera_file = os.path.join(scan_dir, 'undistorted_camera_parameters', f'{scan_id}.conf')
    w, h = 1280, 1024

    try:
        groups = out_group[scan_id]
    except:
        print(f"scan {scan_id} not found in lookup")
        continue

    # group_id, group_nodes = list(groups.items())[0]
    for group_id, group_nodes in groups.items():
        try:
            stream_num = 0
            output_dir = os.path.join(output_root, f"{scan_id}", f"{group_id}")
            # create empty point cloud
            pc = open3d.geometry.PointCloud()
            for r in tqdm(open(camera_file)):
                tokens = r.split()
                if len(tokens)==0:
                    continue
                elif tokens[0]=='intrinsics_matrix':
                    in_mat = list(map(lambda x:float(x), tokens[1:])) #3x3 mat
                    fx, _, cx, _, fy, cy, _, _, _ = in_mat
                    assert len(in_mat)==9
                    in_mat_output = f"{in_mat[0]} 0 {in_mat[2]} 0\n0 {in_mat[4]} {in_mat[5]} 0\n0 0 1 0\n0 0 0 1"
                    in_mat_out_dir = os.path.join(output_dir, "intrinsic")
                    os.makedirs(in_mat_out_dir, exist_ok=True)
                    with open(os.path.join(in_mat_out_dir, f"intrinsic_color.txt"), "w") as f:
                        f.write(in_mat_output)
                    
                elif tokens[0]=='scan':
                    if 'd0' not in tokens[1]: #use d0 only now
                        continue
                    depth_image_file = os.path.join(scan_dir, 'undistorted_depth_images', tokens[1])
                    color_image_file = os.path.join(scan_dir, 'undistorted_color_images', tokens[2])
                    # assert file exists
                    node_name = tokens[1].split('_')[0]
                    if node_name not in group_nodes:
                        continue
                    # resolution = Image.open(color_image_file).size
                    # print(resolution)
                    ex_mat = list(map(lambda x:float(x), tokens[3:])) #4x4 mat
                    ex_mat = np.asarray(ex_mat).reshape(4,4)
                    ex_mat = np.linalg.inv(ex_mat)
                    tfm = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]) # to align with coordinate system of open3d
                    ex_mat = np.dot(tfm, ex_mat)

                    #1. append point cloud
                    depth = open3d.io.read_image(depth_image_file)
                    rgb = open3d.io.read_image(color_image_file)
                    rgbd = open3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth, depth_scale=4000)
                    #pointcloud = open3d.geometry.create_point_cloud_from_rgbd_image(rgbd)
                    intrinsic = open3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
                    pointcloud = open3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic=intrinsic)

                    pointcloud = pointcloud.transform(np.linalg.inv(ex_mat))

                    pc += pointcloud

                    #2. pose
                    ex_mat = ex_mat.reshape(16)
                    ex_mat = [str(x) for x in ex_mat]
                    
                    assert os.path.exists(depth_image_file) and os.path.exists(color_image_file)
                    assert len(ex_mat)==16
                    ex_mat_output = f"{' '.join(ex_mat[:4])}\n{' '.join(ex_mat[4:8])}\n{' '.join(ex_mat[8:12])}\n{' '.join(ex_mat[12:])}"
                    ex_mat_out_dir = os.path.join(output_dir, "pose")
                    os.makedirs(ex_mat_out_dir, exist_ok=True)
                    with open(os.path.join(ex_mat_out_dir, f"{stream_num}.txt"), "w") as f:
                        f.write(ex_mat_output)
                    
                    #3. image links, depth link
                    ### link color image
                    image_out_dir = os.path.join(output_dir, "color")
                    extension = tokens[2].split('.')[-1]
                    os.makedirs(image_out_dir, exist_ok=True)
                    # make absolute path
                    source = os.path.abspath(color_image_file)
                    target = os.path.abspath(os.path.join(image_out_dir, f"{stream_num}.{extension}"))
                    os.system(f"ln -s {source} {target}")

                    ### link depth image
                    depth_out_dir = os.path.join(output_dir, "depth")
                    extension = tokens[1].split('.')[-1]
                    os.makedirs(depth_out_dir, exist_ok=True)
                    # print(depth_image_file)
                    source = os.path.abspath(depth_image_file)  
                    target = os.path.abspath(os.path.join(depth_out_dir, f"{stream_num}.{extension}"))
                    os.system(f"ln -s {source} {target}")
                    stream_num += 1

            
            gt_pc_path = os.path.join(mp3d_rgbd_dir, scan_id, "region_segmentations", f"region{group_id}.ply")
            gt_pc_path = os.path.abspath(gt_pc_path)
            target_path = os.path.join(output_dir, "pointcloud.ply")
            assert os.path.exists(gt_pc_path)
            # link
            os.system(f"ln -s {gt_pc_path} {output_dir}/pointcloud.ply")
        except Exception as e:
            print(e) 
            print(f"scan {scan_id} group {group_id} failed")
            continue

