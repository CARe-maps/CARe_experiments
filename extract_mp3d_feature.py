import glob
import os


resource_dir = "./resources/mp3d_output"
output_dir = "./extracted_features/mp3d_output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# get_tasks
tasks = glob.glob(f"{resource_dir}/*/*/") 

for task in sorted(tasks):
    task = task[:-1]
    print(f"Processing {task}")
    temp = task.split("/")
    scan_id = temp[-1]
    region_id = temp[-2]
    out_dir = f"{output_dir}/{region_id}/{scan_id}"
    # use absolute path
    task = os.path.abspath(task)
    out_dir = os.path.abspath(out_dir)
    # run this in system: bash roomwise_openmask3d_worker.sh task out_dir
    result = os.system(f"bash roomwise_openmask3d_worker.sh {task} {out_dir}")

    print(f"Finished {task}")