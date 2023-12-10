from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download
from huggingface_hub import login


login()


# x = hf_hub_download(repo_id="ShapeNet/ShapeNetCore", filename="04554684.zip")
# x = hf_hub_download(repo_id="tiange/Cap3D", filename="Cap3D_automated_Objaverse_highquality.csv")
# print(x)


# x = snapshot_download(repo_id="ShapeNet/shapenetcore-glb", repo_type='dataset', cache_dir='/root/shapenet')
# x = snapshot_download(repo_id="tiange/Cap3D", repo_type='dataset', local_dir="shape_data",local_dir_use_symlinks=False)
x = snapshot_download(repo_id="tiange/Cap3D", repo_type='dataset',filename="Cap3D_automated_Objaverse_highquality.csv",local_dir_use_symlinks=False)

print(x)