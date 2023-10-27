from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download
from huggingface_hub import login


login()


# x = hf_hub_download(repo_id="runwayml/stable-diffusion-v1-5", filename="v1-5-pruned.ckpt")
# print(x)


x = snapshot_download(repo_id="ShapeNet/shapenetcore-glb")
print(x)