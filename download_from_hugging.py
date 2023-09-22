from huggingface_hub import hf_hub_download

x = hf_hub_download(repo_id="lllyasviel/ControlNet-v1-1", filename="control_v11p_sd15_canny.pth")
print(x)