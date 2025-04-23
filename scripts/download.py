from huggingface_hub import snapshot_download

# Repository ID and local directory
repo_id = "alignment-handbook/zephyr-7b-sft-full" 
local_dir = "model_hub/zephyr-7b-sft-full"  # Specify your desired local directory

# Download the model repository
print(f"Downloading model from {repo_id} to {local_dir}...")
snapshot_download(
    repo_id=repo_id,
    repo_type="model",  # Specify repo type as 'model'
    local_dir=local_dir,  # Specify the local directory
    local_dir_use_symlinks=False,  # Avoid symlinks for complete local copy
)

print(f"Model successfully downloaded to {local_dir}!")

###################################################################
