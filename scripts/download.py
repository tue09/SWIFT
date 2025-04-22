from huggingface_hub import snapshot_download

# Repository ID and local directory
repo_id = "Qwen/Qwen1.5-1.8B" 
local_dir = "model_hub/Qwen1.5-1.8B_base"  # Specify your desired local directory

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

repo_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 
local_dir = "model_hub/TinyLlama-1.1B_base"  # Specify your desired local directory

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
