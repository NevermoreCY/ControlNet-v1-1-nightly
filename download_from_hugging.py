from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download
from huggingface_hub import login


login()


# x = hf_hub_download(repo_id="ShapeNet/ShapeNetCore", filename="04554684.zip")
# print(x)


x = snapshot_download(repo_id="ShapeNet/shapenetcore-glb", repo_type='dataset')
print(x)