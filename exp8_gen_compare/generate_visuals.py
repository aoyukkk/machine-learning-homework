import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import DDPMPipeline
import os

os.makedirs("figures", exist_ok=True)

try:
    pipe = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32").to("cuda")
    scheduler = pipe.scheduler
    noise_init = torch.randn((1, 3, 32, 32)).to("cuda")
    image = noise_init.clone()
    scheduler.set_timesteps(1000)
    steps_to_save = np.linspace(0, 999, 8, dtype=int)
    saved_diff = []
    
    for i, t in enumerate(scheduler.timesteps):
        with torch.no_grad():
            model_output = pipe.unet(image, t).sample
        image = scheduler.step(model_output, t, image).prev_sample
        if i in steps_to_save or i == 999:
            out = (image / 2 + 0.5).clamp(0, 1).squeeze().cpu().permute(1, 2, 0).numpy()
            saved_diff.append(out)
            
    fig, axes = plt.subplots(1, 8, figsize=(16, 2.2))
    title_steps = ["Noise"] + [f"Step {s}/7" for s in range(1, 8)]
    for ax, img, title in zip(axes, saved_diff, title_steps):
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig("figures/diffusion_process.png", dpi=150)
    plt.close()

    final_img = saved_diff[-1]
    saved_ar = []
    fractions = [0.0, 0.15, 0.3, 0.5, 0.75, 0.9, 1.0]
    for frac in fractions:
        ar_img = np.zeros_like(final_img)
        limit = int(frac * 32 * 32)
        r, c = limit // 32, limit % 32
        if r > 0: ar_img[:r, :, :] = final_img[:r, :, :]
        if r < 32: ar_img[r, :c, :] = final_img[r, :c, :]
        saved_ar.append(ar_img)
        
    fig, axes = plt.subplots(1, 7, figsize=(14, 2.2))
    for i, (ax, img) in enumerate(zip(axes, saved_ar)):
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"Gen {int(fractions[i]*100)}%", fontsize=12)
    plt.tight_layout()
    plt.savefig("figures/ar_process.png", dpi=150)
    plt.close()
    print("Successfully generated process visuals.")
except Exception as e:
    print(f"Error visualizing: {e}")
