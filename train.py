from utils.models import VGGEncoder
import argparse
import os
import torch
import torch.nn as nn 
from pathlib import Path
from PIL import Image
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.utils import *
from utils.models import *
import torch.optim as optim 
from tqdm import tqdm 
from torchvision.utils import save_image    


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_dir",type = str, default=r"C:\Major projects\NST-Neural Style Transfer\content_data_test", help="Location of Content Dataset")
    parser.add_argument("--style_dir",type = str, default=r"C:\Major projects\NST-Neural Style Transfer\style_data_test", help="Location of Style Dataset")
    parser.add_argument("--vgg",type = str, default=r"C:\Major projects\NST-Neural Style Transfer\vgg_normalised.pth", help="Path to VGG model")
    parser.add_argument("--experiment",type = str, default="experiment_1", help="Experiment name")
    parser.add_argument("--final_size", type = int, default= 256, help = "Size of final output image")
    parser.add_argument("--content_img_size", type = int, default= 128, help = "Size of content images (reduced for 2GB GPU)")
    parser.add_argument("--style_img_size", type = int, default= 128, help = "Size of style images (reduced for 2GB GPU)")
    parser.add_argument("--crop", type = bool, default= True, help = "Crop the image")
    parser.add_argument("--batch_size", type = int, default= 4, help = "Batch Size (kept at 4 for 2GB GPU)")
    parser.add_argument("--lr", type = float, default= 1e-4, help = "Learning Rate")
    parser.add_argument("--lr_decay", type = float, default= 5e-5, help = "Learning Rate Decay")
    parser.add_argument("--epochs", type = int, default= 2, help = "Number of Epochs")
    parser.add_argument("--content_weight", type = float, default= 1.0, help = "Content Weight")
    parser.add_argument("--style_weight", type = float, default= 1.0, help = "Style Weight")
    parser.add_argument("--loss_interval", type = int, default= 10, help = "Print Loss after every N epochs")
    parser.add_argument("--save_interval", type = int, default= 1, help = "Save model checkpoint every N epochs")
    parser.add_argument("--decoder_path", type = str, default= None, help = "decoder model path")
    parser.add_argument("--optimizer_path", type = str, default= None, help = "optimizer model path")
    parser.add_argument("--resume", type = bool, default= False, help = "Resume the model")

    return parser.parse_args()

    

def main():
    args = parse_arguments()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Print GPU info for debugging
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Using GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        # Clear any leftover GPU memory from previous runs
        torch.cuda.empty_cache()

    save_dir = Path("experiments") / args.experiment
    save_dir.mkdir(exist_ok = True, parents = True)

    # Save argument Values
    with open(save_dir / "args.txt", "w") as f:
        for arg, value in args.__dict__.items():
            f.write(f"{arg}:{value}\n")     
    
    content_transforms = get_transform(args.content_img_size,args.crop, args.final_size)
    style_transforms = get_transform(args.style_img_size,args.crop, args.final_size)

    content_dataset = ImageFolderDataset(root = args.content_dir, transform= content_transforms) 
    style_dataset = ImageFolderDataset(args.style_dir, style_transforms) 

    # num_workers=0 on Windows to avoid shared memory issues; pin_memory only when using CUDA
    content_dataloader = DataLoader(content_dataset, batch_size=args.batch_size, shuffle= True, 
                                    pin_memory=(device == "cuda"), drop_last=True, num_workers=0)

    style_dataloader = DataLoader(style_dataset, batch_size=args.batch_size, shuffle= True, 
                                   pin_memory=(device == "cuda"), drop_last=True, num_workers=0)
    


    vgg_encoder = VGGEncoder(args.vgg).to(device)
    decoder = Decoder().to(device)

    
    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)

    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda= lambda epoch: 1.0 /(1.0 + args.lr_decay * epoch)
    )
    
    if args.resume:
        decoder.load_state_dict(torch.load(args.decoder_path))
        optimizer.load_state_dict(torch.load(args.optimizer_path))
    
    print(f"Training started with batch_size={args.batch_size}, "
          f"content_size={args.content_img_size}, style_size={args.style_img_size}")
    
    mse_loss = nn.MSELoss()

    epochs = args.epochs

    vgg_encoder.eval()

    
    total_train_loss = 0.0 
    total_closs = 0.0
    total_sloss = 0.0

    for epoch in range(epochs):
        # Clear GPU cache at the start of each epoch
        if device == "cuda":
            torch.cuda.empty_cache()

        running_loss = 0.0
        running_content_loss = 0.0
        running_style_loss = 0.0


        progress_bar = tqdm(zip(content_dataloader, style_dataloader), total=min(len(content_dataloader), len(style_dataloader)))

        for content_batch, style_batch in progress_bar:

            content_batch = content_batch.to(device, non_blocking=True)
            style_batch = style_batch.to(device, non_blocking=True)

            # Free memory from previous iteration's gradients
            optimizer.zero_grad(set_to_none=True)

            # Content & style features don't need gradients (encoder is frozen)
            # but g_feats below MUST keep gradients for decoder backward pass
            with torch.no_grad():
                c_feats = vgg_encoder(content_batch)
                s_feats = vgg_encoder(style_batch)

            t = adaptive_instance_normalization(c_feats[-1], s_feats[-1]) 

            # Free input images from GPU — we only need features from here on
            # Keep CPU copies of the last batch for saving sample outputs
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

            progress_bar.set_description(f"Loss : {loss.item():4f} | Content Loss: {loss_c.item():4f} | Style Loss: {loss_s.item():4f}")

            running_loss += loss.item()
            running_content_loss += loss_c.item()
            running_style_loss += loss_s.item()

        total_train_loss += running_loss / len(content_dataloader)
        total_closs += running_content_loss / len(content_dataloader)
        total_sloss += running_style_loss / len(content_dataloader)

        scheduler.step()

        if (epoch + 1) % args.loss_interval == 0 :
            tqdm.write(f"For epochs {epoch+1}/{epochs} Total training loss is : {total_train_loss / (epoch+1):4f} Content Loss : {total_closs / (epoch+1):4f} Style Loss : {total_sloss / (epoch+1):4f}")
            

        if (epoch+1) % args.save_interval == 0 :
            torch.save(decoder.state_dict(), save_dir / f"decoder_epoch_{epoch+1}.pth")
            torch.save(optimizer.state_dict(), save_dir / f"optimizer_state_{epoch+1}.pth" )
            
            with torch.no_grad():
                output = torch.cat([last_content, last_style, g.detach().cpu()], dim = 0)
                save_image(output, save_dir / f"output_{epoch+1}.png", nrow = args.batch_size)


if __name__ == "__main__":
    main()
