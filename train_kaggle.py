"""
Kaggle-optimized training script for AdaIN Neural Style Transfer.
Designed for 16GB GPU (T4/P100) with best model tracking.

Usage on Kaggle:
    !python train_kaggle.py --content_dir="/path/to/content" --style_dir="/path/to/style" \
        --vgg="/path/to/vgg_normalised.pth" --experiment="kaggle_run_1" --epochs=20
"""

import argparse
import os
import csv
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image, ImageFile
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transform
from torchvision.utils import save_image
from tqdm import tqdm
import random

# Allow loading of truncated image files (common in large datasets)
ImageFile.LOAD_TRUNCATED_IMAGES = True
# Disable DecompressionBombError for massive high-res images
Image.MAX_IMAGE_PIXELS = None


# ============================================================
# Models (inlined to avoid import issues on Kaggle)
# ============================================================

class VGGEncoder(nn.Module):
    def __init__(self, vgg_path):
        super(VGGEncoder, self).__init__()
        self.vgg = nn.Sequential(
            nn.Conv2d(3, 3, (1, 1)),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(3, 64, (3, 3)),
            nn.ReLU(),  # relu1-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),  # relu1-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(),  # relu2-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),  # relu2-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 256, (3, 3)),
            nn.ReLU(),  # relu3-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-4
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, (3, 3)),
            nn.ReLU(),  # relu4-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-4
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU()   # relu5-4
        )
        self.vgg.load_state_dict(torch.load(vgg_path, weights_only=True))
        self.vgg = nn.Sequential(*list(self.vgg.children())[:31])
        enc_layers = list(self.vgg.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])
        self.enc_2 = nn.Sequential(*enc_layers[4:11])
        self.enc_3 = nn.Sequential(*enc_layers[11:18])
        self.enc_4 = nn.Sequential(*enc_layers[18:31])

        for name in ["enc_1", "enc_2", "enc_3", "enc_4"]:
            for p in getattr(self, name).parameters():
                p.requires_grad = False

    def forward(self, x, is_test=False):
        h1 = self.enc_1(x)
        h2 = self.enc_2(h1)
        h3 = self.enc_3(h2)
        h4 = self.enc_4(h3)
        if is_test:
            return h4
        return [h1, h2, h3, h4]


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, (3, 3)),
        )

    def forward(self, x):
        return self.decoder(x)


# ============================================================
# Utility Functions
# ============================================================

class ImageFolderDataset(Dataset):
    def __init__(self, root, transform=None):
        super(ImageFolderDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.files = [p for p in os.listdir(root) if p.lower().endswith((".jpg", ".jpeg", ".png"))]
        print(f"  Found {len(self.files)} images in {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            img_path = os.path.join(self.root, self.files[idx])
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img
        except (OSError, IOError):
            return self.__getitem__(random.randint(0, len(self.files) - 1))


def get_transform(img_size, crop=False, final_size=512):
    transform_list = []
    if img_size > 0:
        transform_list.append(transform.Resize(img_size))
    if crop:
        transform_list.append(transform.RandomCrop(img_size))
    else:
        transform_list.append(transform.RandomCrop(final_size))
    transform_list.append(transform.ToTensor())
    return transform.Compose(transform_list)


def adaptive_instance_normalization(content_feats, style_feats):
    size = content_feats.size()
    style_mean, style_std = compute_mean_std(style_feats)
    content_mean, content_std = compute_mean_std(content_feats)
    normalized = (content_feats - content_mean.expand(size)) / content_std.expand(size)
    return normalized * style_std.expand(size) + style_mean.expand(size)


def compute_mean_std(feats, eps=1e-5):
    size = feats.size()
    assert len(size) == 4
    mean = torch.mean(feats, dim=[2, 3], keepdim=True)
    std = torch.std(feats, dim=[2, 3], keepdim=True)
    return mean, std + eps


# ============================================================
# Training
# ============================================================

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_dir", type=str, required=True, help="Content dataset path")
    parser.add_argument("--style_dir", type=str, required=True, help="Style dataset path")
    parser.add_argument("--vgg", type=str, required=True, help="Path to vgg_normalised.pth")
    parser.add_argument("--experiment", type=str, default="kaggle_run_1", help="Experiment name")
    parser.add_argument("--content_img_size", type=int, default=256, help="Content image size")
    parser.add_argument("--style_img_size", type=int, default=256, help="Style image size")
    parser.add_argument("--final_size", type=int, default=256, help="Final output size")
    parser.add_argument("--crop", type=bool, default=True, help="Random crop")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (8 for 16GB GPU)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lr_decay", type=float, default=5e-5, help="LR decay")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--content_weight", type=float, default=1.0, help="Content loss weight")
    parser.add_argument("--style_weight", type=float, default=1.0, help="Style loss weight")
    parser.add_argument("--loss_interval", type=int, default=1, help="Print loss every N epochs")
    parser.add_argument("--save_interval", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--decoder_path", type=str, default=None, help="Resume decoder path")
    parser.add_argument("--optimizer_path", type=str, default=None, help="Resume optimizer path")
    parser.add_argument("--resume", type=bool, default=False, help="Resume training")
    return parser.parse_args()


def main():
    args = parse_arguments()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Using GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        torch.cuda.empty_cache()
    else:
        print("WARNING: No GPU detected! Training will be very slow.")

    save_dir = Path("experiments") / args.experiment
    save_dir.mkdir(exist_ok=True, parents=True)

    # Save arguments
    with open(save_dir / "args.txt", "w") as f:
        for arg, value in args.__dict__.items():
            f.write(f"{arg}: {value}\n")

    # CSV log for loss tracking
    csv_path = save_dir / "training_log.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "avg_total_loss", "avg_content_loss", "avg_style_loss", "lr", "is_best"])

    # Transforms & Datasets
    print("\nLoading datasets...")
    content_transforms = get_transform(args.content_img_size, args.crop, args.final_size)
    style_transforms = get_transform(args.style_img_size, args.crop, args.final_size)

    content_dataset = ImageFolderDataset(root=args.content_dir, transform=content_transforms)
    style_dataset = ImageFolderDataset(args.style_dir, style_transforms)

    content_dataloader = DataLoader(
        content_dataset, batch_size=args.batch_size, shuffle=True,
        pin_memory=True, drop_last=True, num_workers=args.num_workers
    )
    style_dataloader = DataLoader(
        style_dataset, batch_size=args.batch_size, shuffle=True,
        pin_memory=True, drop_last=True, num_workers=args.num_workers
    )

    # Models
    print("\nLoading models...")
    vgg_encoder = VGGEncoder(args.vgg).to(device)
    decoder = Decoder().to(device)

    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: 1.0 / (1.0 + args.lr_decay * epoch)
    )

    if args.resume and args.decoder_path:
        decoder.load_state_dict(torch.load(args.decoder_path, weights_only=True))
        if args.optimizer_path:
            optimizer.load_state_dict(torch.load(args.optimizer_path, weights_only=True))
        print("Resumed from checkpoint")

    batches_per_epoch = min(len(content_dataloader), len(style_dataloader))
    total_iterations = batches_per_epoch * args.epochs
    print(f"\n{'='*60}")
    print(f"Training Config:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Image size: {args.content_img_size}x{args.content_img_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batches/epoch: {batches_per_epoch}")
    print(f"  Total iterations: {total_iterations:,}")
    print(f"  Content weight: {args.content_weight} | Style weight: {args.style_weight}")
    print(f"{'='*60}\n")

    mse_loss = nn.MSELoss()
    vgg_encoder.eval()

    # ---- Best model tracking ----
    best_loss = float("inf")

    for epoch in range(args.epochs):
        if device == "cuda":
            torch.cuda.empty_cache()

        running_loss = 0.0
        running_content_loss = 0.0
        running_style_loss = 0.0

        progress_bar = tqdm(
            zip(content_dataloader, style_dataloader),
            total=batches_per_epoch,
            desc=f"Epoch {epoch+1}/{args.epochs}"
        )

        for content_batch, style_batch in progress_bar:
            content_batch = content_batch.to(device, non_blocking=True)
            style_batch = style_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                c_feats = vgg_encoder(content_batch)
                s_feats = vgg_encoder(style_batch)

            t = adaptive_instance_normalization(c_feats[-1], s_feats[-1])

            last_content = content_batch.detach().cpu()
            last_style = style_batch.detach().cpu()
            del content_batch, style_batch

            g = decoder(t)
            g_feats = vgg_encoder(g)

            loss_c = mse_loss(g_feats[-1], t) * args.content_weight
            loss_s = 0.0
            for g_f, s_f in zip(g_feats, s_feats):
                g_mean, g_std = compute_mean_std(g_f)
                s_mean, s_std = compute_mean_std(s_f)
                loss_s += (mse_loss(g_mean, s_mean) + mse_loss(g_std, s_std)) * args.style_weight

            loss = loss_c + loss_s
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix({
                "loss": f"{loss.item():.2f}",
                "c_loss": f"{loss_c.item():.2f}",
                "s_loss": f"{loss_s.item():.2f}"
            })

            running_loss += loss.item()
            running_content_loss += loss_c.item()
            running_style_loss += loss_s.item()

        # Epoch-level metrics
        avg_loss = running_loss / batches_per_epoch
        avg_closs = running_content_loss / batches_per_epoch
        avg_sloss = running_style_loss / batches_per_epoch
        current_lr = optimizer.param_groups[0]["lr"]

        scheduler.step()

        # ---- Check for best model ----
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
            torch.save(decoder.state_dict(), save_dir / "best_decoder.pth")
            torch.save(optimizer.state_dict(), save_dir / "best_optimizer.pth")

        # Log to CSV
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, f"{avg_loss:.6f}", f"{avg_closs:.6f}", f"{avg_sloss:.6f}", f"{current_lr:.8f}", is_best])

        # Print epoch summary
        if (epoch + 1) % args.loss_interval == 0:
            best_marker = " *** NEW BEST ***" if is_best else ""
            tqdm.write(
                f"Epoch {epoch+1}/{args.epochs} | "
                f"Loss: {avg_loss:.4f} | Content: {avg_closs:.4f} | Style: {avg_sloss:.4f} | "
                f"LR: {current_lr:.6f}{best_marker}"
            )

        # Save checkpoint & sample output
        if (epoch + 1) % args.save_interval == 0:
            torch.save(decoder.state_dict(), save_dir / f"decoder_epoch_{epoch+1}.pth")
            torch.save(optimizer.state_dict(), save_dir / f"optimizer_state_{epoch+1}.pth")

            with torch.no_grad():
                output = torch.cat([last_content, last_style, g.detach().cpu()], dim=0)
                save_image(output, save_dir / f"output_{epoch+1}.png", nrow=args.batch_size)

    # ---- Final Summary ----
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"   Best loss: {best_loss:.4f}")
    print(f"   Best model saved to: {save_dir / 'best_decoder.pth'}")
    print(f"   Training log saved to: {csv_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
