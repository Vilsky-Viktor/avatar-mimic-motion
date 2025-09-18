import torch
from torch.utils.data import DataLoader
from diffusers import UNet2DConditionModel, DDPMScheduler
from peft import LoraConfig, get_peft_model
from pathlib import Path
from avatar_training_job.dataloader import PoseImageDataset
import logging

log = logging.getLogger("trainer")

def train_lora(unet_dir: Path, photos_dir: Path, poses_dir: Path, out_dir: Path,
               num_epochs: int = 5, batch_size: int = 4, lr: float = 1e-4):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    unet = UNet2DConditionModel.from_pretrained(str(unet_dir))
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.1,
        bias="none",
        task_type="UNET"
    )
    unet = get_peft_model(unet, config).to(device)

    noise_scheduler = DDPMScheduler.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt-1-1")
    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr)

    dataset = PoseImageDataset(photos_dir, poses_dir, image_size=256)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    log.info("Training...")
    for epoch in range(num_epochs):
        for images, poses in dataloader:
            images, poses = images.to(device), poses.to(device)

            noise = torch.randn_like(images)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                      (images.shape[0],), device=device)

            noisy_latents = noise_scheduler.add_noise(images, noise, timesteps)
            pred = unet(noisy_latents, timesteps, encoder_hidden_states=poses).sample

            loss = torch.nn.functional.mse_loss(pred, noise)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        log.info(f"Epoch {epoch+1}/{num_epochs} - loss={loss.item():.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    unet.save_pretrained(str(out_dir))
    log.info(f"Saved LoRA to {out_dir}")