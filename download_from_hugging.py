from huggingface_hub import hf_hub_download

x = hf_hub_download(repo_id="runwayml/stable-diffusion-v1-5", filename="v1-5-pruned.ckpt")
print(x)