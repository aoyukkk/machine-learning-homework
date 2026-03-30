import torch
from diffusers import DDPMPipeline
import time

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading DDPM CIFAR10 pipeline on {device}...")
    pipeline = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32")
    pipeline = pipeline.to(device)

    print("Generating 64 samples for visual comparison...")
    start_time = time.time()
    images = pipeline(batch_size=64, num_inference_steps=1000).images
    end_time = time.time()

    print(f"Time taken for 64 samples: {end_time - start_time:.2f} seconds")

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(8, 8, figsize=(8, 8))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(images[i])
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("figures/sota_samples.png")
    print("Saved samples to figures/sota_samples.png")

if __name__ == "__main__":
    main()
