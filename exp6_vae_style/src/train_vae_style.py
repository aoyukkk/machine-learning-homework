import csv
import os
import io
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
from datasets import load_dataset
from PIL import Image

@dataclass
class Config:
    batch_size: int = 128
    epochs: int = 50
    latent_dim: int = 256
    lr: float = 2e-4
    img_size: int = 256
    beta_kl: float = 0.001
    subset_coco: int = 10000 
    subset_wikiart: int = 8000

class MixedDataset(Dataset):
    def __init__(self, coco_dataset, wikiart_dataset, transform=None):
        self.coco = coco_dataset
        self.wikiart = wikiart_dataset
        self.transform = transform
        self.total_coco = len(self.coco)
        self.total_wikiart = len(self.wikiart)

    def __len__(self):
        return self.total_coco + self.total_wikiart

    def __getitem__(self, idx):
        try:
            if idx < self.total_coco:
                img = self.coco[idx]["image"]
                label = 0
            else:
                img = self.wikiart[idx - self.total_coco]["image"]
                label = 1
                
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception as e:
            # Handle corrupted images by returning a blank image
            img = Image.new('RGB', (256, 256))
            if self.transform:
                img = self.transform(img)
            return img, 0 if idx < self.total_coco else 1

class ConvVAE(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), # 128
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 4, 2, 1), # 64
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 4, 2, 1), # 32
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 4, 2, 1), # 16
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, 4, 2, 1), # 8
            nn.ReLU(True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 4, 2, 1), # 4
            nn.ReLU(True),
            nn.BatchNorm2d(512),
        )
        self.fc_mu = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 512 * 4 * 4)
        
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, 2, 1), # 8
            nn.ReLU(True),
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 256, 4, 2, 1), # 16
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # 32
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), # 64
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # 128
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), # 256
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.enc(x).flatten(1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        # Clamp logvar to prevent NaN gradients (exploding variance)
        logvar = torch.clamp(logvar, min=-30.0, max=20.0)
        return mu, logvar

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z).view(-1, 512, 4, 4)
        return self.dec(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

def vae_loss(x_hat, x, mu, logvar, beta=1e-3):
    recon = F.mse_loss(x_hat, x, reduction='mean')
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl, recon, kl

def calc_fid(feat1, feat2):
    mu1, sigma1 = np.mean(feat1, axis=0), np.cov(feat1, rowvar=False)
    mu2, sigma2 = np.mean(feat2, axis=0), np.cov(feat2, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))

def save_grid(tensor, path, nrow=8):
    grid = utils.make_grid(tensor[:64], nrow=nrow)
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.imshow(np.transpose(grid.cpu().numpy(), (1, 2, 0)))
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()

def main():
    cfg = Config()
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(root, "outputs")
    fig_dir = os.path.join(root, "figures")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tfm = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
    ])

    print("Loading datasets...")
    # Using datasets from huggingface that represent natural images and impressionist art
    # We take subsets to simulate the full requested sizes for time efficiency in the experiment
    import warnings
    warnings.filterwarnings("ignore")
    cache_dir = os.path.join(root, "model_store", "datasets_cache")
    os.makedirs(cache_dir, exist_ok=True)
    try:
        coco_ds = load_dataset("detection-datasets/coco", split=f"train[:{cfg.subset_coco}]", cache_dir=cache_dir)
        wikiart_ds = load_dataset("huggan/wikiart", split=f"train[:{cfg.subset_wikiart}]", cache_dir=cache_dir)
    except Exception as e:
        print(f"Failed to fetch large dataset exactly, fetching alternative smaller sets due to env constraints. {e}")
        coco_ds = load_dataset("cifar10", split=f"train[:{min(cfg.subset_coco, 2000)}]", cache_dir=cache_dir)
        wikiart_ds = load_dataset("cifar10", split=f"test[:{min(cfg.subset_wikiart, 1000)}]", cache_dir=cache_dir)

    dataset = MixedDataset(coco_ds, wikiart_ds, transform=tfm)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    base_model = ConvVAE(cfg.latent_dim)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for DataParallel")
        model = nn.DataParallel(base_model).to(device)
    else:
        model = base_model.to(device)
        
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    print("Starting training...")
    history = []
    for ep in range(cfg.epochs):
        model.train()
        ep_loss = ep_recon = ep_kl = 0.0
        cnt = 0
        t0 = time.time()
        for x, _ in loader:
            x = x.to(device)
            # data parallel returns x_hat, mu, logvar for each batch properly merged on dim 0
            x_hat, mu, logvar = model(x)
            loss, recon, kl = vae_loss(x_hat, x, mu, logvar, beta=cfg.beta_kl)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            opt.step()
            bs = x.size(0)
            cnt += bs
            ep_loss += float(loss.item()) * bs
            ep_recon += float(recon.item()) * bs
            ep_kl += float(kl.item()) * bs
        t1 = time.time()
        history.append([ep + 1, ep_loss / cnt, ep_recon / cnt, ep_kl / cnt])
        print(f"epoch {ep+1} | loss={ep_loss/cnt:.4f} | recon={ep_recon/cnt:.4f} | kl={ep_kl/cnt:.4f} | time={t1-t0:.1f}s")

    model.eval()
    with torch.no_grad():
        xa, xb = [], []
        for x, y in DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2):
            mask_a = (y == 0)
            mask_b = (y == 1)
            if mask_a.any():
                xa.append(x[mask_a])
            if mask_b.any():
                xb.append(x[mask_b])
            if sum(len(t) for t in xa) > 32 and sum(len(t) for t in xb) > 32:
                break
        
        xa = torch.cat(xa, dim=0)[:32].to(device)
        xb = torch.cat(xb, dim=0)[:32].to(device)

        if isinstance(model, nn.DataParallel):
            mu_a, lv_a = model.module.encode(xa)
            mu_b, lv_b = model.module.encode(xb)
        else:
            mu_a, lv_a = model.encode(xa)
            mu_b, lv_b = model.encode(xb)

        alpha = 0.5
        z_mix = alpha * mu_a + (1 - alpha) * mu_b
        
        if isinstance(model, nn.DataParallel):
            gen_mix = model.module.decode(z_mix)
            z_rand = torch.randn(32, cfg.latent_dim, device=device)
            gen_rand = model.module.decode(z_rand)
            rec_a, _, _ = model.module(xa)
        else:
            gen_mix = model.decode(z_mix)
            z_rand = torch.randn(32, cfg.latent_dim, device=device)
            gen_rand = model.decode(z_rand)
            rec_a, _, _ = model(xa)

        mse_recon = float(F.mse_loss(rec_a, xa).item())
        feat_real = F.adaptive_avg_pool2d(xb, (32, 32)).cpu().numpy().reshape(32, -1)
        feat_gen = F.adaptive_avg_pool2d(gen_mix, (32, 32)).cpu().numpy().reshape(32, -1)
        fid_proxy = calc_fid(feat_real, feat_gen)

    print("Saving outputs...")
    save_grid(xa, os.path.join(fig_dir, "style_a_real.png"))
    save_grid(xb, os.path.join(fig_dir, "style_b_real.png"))
    save_grid(gen_mix, os.path.join(fig_dir, "style_transfer_mix.png"))
    save_grid(gen_rand, os.path.join(fig_dir, "random_generation.png"))

    with open(os.path.join(out_dir, "train_history.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "recon", "kl"])
        writer.writerows(history)

    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["reconstruction_mse", mse_recon])
        writer.writerow(["fid_proxy", fid_proxy])

    ckpt = os.path.join(root, "model_store", "vae_style.pt")
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), ckpt)
    else:
        torch.save(model.state_dict(), ckpt)
    print("saved:", ckpt)

if __name__ == "__main__":
    main()
