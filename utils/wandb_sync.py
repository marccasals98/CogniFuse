import wandb
import torch
import os 

wandb_dir = "/gpfs/projects/bsc88/speech/speaker_recognition/outputs/ser_2025/wandb"
run_id = "c295a5hc"
checkpoint_folder = "/gpfs/projects/bsc88/speech/speaker_recognition/outputs/ser_2025/models/25_01_10_12_54_18_WAV2VEC2_XLSR_300M_BERT_LARGE_UNCASED_NoneAdapter_NoneAdapter_MultiHeadAttention_AttentionPooling_c295a5hc"
checkpoint_name = "25_01_10_12_54_18_WAV2VEC2_XLSR_300M_BERT_LARGE_UNCASED_NoneAdapter_NoneAdapter_MultiHeadAttention_AttentionPooling_c295a5hc.chkpt"

checkpoint_path = os.path.join(checkpoint_folder, checkpoint_name)
checkpoint = torch.load(checkpoint_path, weights_only=False)

wandb_run = wandb.init(
    project = "emotions_trains_2025", 
    job_type = "training", 
    entity = "upc-veu",
    dir = wandb_dir,
    id = run_id,
    resume = "allow",
    )

print(f"wandb running online/offline: {wandb_run.settings.mode}")
print(f"wandb_dir: {wandb_dir}")
print(f"id_name : {wandb_run.id}_{wandb_run.name}")



print(f'Starting to save checkpoint as wandb artifact...')

# Define the artifact
trained_model_artifact = wandb.Artifact(
    name = checkpoint["settings"].model_name,
    type = "trained_model",
    description = checkpoint["settings"].model_architecture_name,
    metadata = vars(checkpoint["settings"]),
)

# Add folder directory
#checkpoint_folder = os.path.join(self.params.model_output_folder, self.params.model_name)
print(f'checkpoint_folder {checkpoint_folder}')
trained_model_artifact.add_dir(
    local_path = checkpoint_folder,
    skip_cache = True,
)

# Log the artifact
wandb.run.log_artifact(trained_model_artifact)

print(f'Artifact saved.')

wandb_run.config.update(vars(checkpoint["settings"]))
    

    

    

    