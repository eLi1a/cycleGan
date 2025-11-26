# train.py
import argparse
from pathlib import Path
import itertools

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.utils as vutils

from datasets import UnpairedImageDataset
from models import ResnetGenerator, NLayerDiscriminator, init_weights


def denorm(x):
    # [-1,1] -> [0,1]
    return (x + 1) / 2.0


def train(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # data
    dataset = UnpairedImageDataset(opt.dataroot, size=opt.size)
    loader = DataLoader(dataset, batch_size=opt.batch_size,
                        shuffle=True, num_workers=opt.num_workers, pin_memory=True)

    # models
    G_A2B = ResnetGenerator().to(device)
    G_B2A = ResnetGenerator().to(device)
    D_A = NLayerDiscriminator().to(device)
    D_B = NLayerDiscriminator().to(device)

    for net in [G_A2B, G_B2A, D_A, D_B]:
        init_weights(net)

    # losses
    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_id = nn.L1Loss()

    # optimizers
    opt_G = optim.Adam(
        itertools.chain(G_A2B.parameters(), G_B2A.parameters()),
        lr=opt.lr, betas=(0.5, 0.999)
    )
    opt_D_A = optim.Adam(D_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    opt_D_B = optim.Adam(D_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    # simple lr decay
    def lambda_rule(epoch):
        if epoch < opt.n_epochs:
            return 1.0
        else:
            return 1.0 - float(epoch - opt.n_epochs) / float(max(1, opt.n_epochs_decay))

    sched_G = optim.lr_scheduler.LambdaLR(opt_G, lr_lambda=lambda_rule)
    sched_D_A = optim.lr_scheduler.LambdaLR(opt_D_A, lr_lambda=lambda_rule)
    sched_D_B = optim.lr_scheduler.LambdaLR(opt_D_B, lr_lambda=lambda_rule)

    out_root = Path(opt.out_dir)
    samples_dir = out_root / "samples"
    ckpt_dir = out_root / "checkpoints"
    samples_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def real_like(pred):
        return torch.ones_like(pred, device=device)

    def fake_like(pred):
        return torch.zeros_like(pred, device=device)

    total_epochs = opt.n_epochs + opt.n_epochs_decay
    print("total epochs:", total_epochs)

    for epoch in range(1, total_epochs + 1):
        for i, batch in enumerate(loader):
            real_A = batch["A"].to(device)
            real_B = batch["B"].to(device)

            # --------- train G ---------
            opt_G.zero_grad()

            # identity
            idt_B = G_A2B(real_B)
            idt_A = G_B2A(real_A)
            loss_idt_B = criterion_id(idt_B, real_B) * opt.lambda_id * opt.lambda_cycle
            loss_idt_A = criterion_id(idt_A, real_A) * opt.lambda_id * opt.lambda_cycle

            # GAN
            fake_B = G_A2B(real_A)
            pred_fake_B = D_B(fake_B)
            loss_G_A2B = criterion_GAN(pred_fake_B, real_like(pred_fake_B))

            fake_A = G_B2A(real_B)
            pred_fake_A = D_A(fake_A)
            loss_G_B2A = criterion_GAN(pred_fake_A, real_like(pred_fake_A))

            # cycle
            rec_A = G_B2A(fake_B)
            rec_B = G_A2B(fake_A)
            loss_cycle_A = criterion_cycle(rec_A, real_A) * opt.lambda_cycle
            loss_cycle_B = criterion_cycle(rec_B, real_B) * opt.lambda_cycle

            loss_G = (loss_G_A2B + loss_G_B2A +
                      loss_cycle_A + loss_cycle_B +
                      loss_idt_A + loss_idt_B)
            loss_G.backward()
            opt_G.step()

            # --------- train D_A ---------
            opt_D_A.zero_grad()
            pred_real_A = D_A(real_A)
            pred_fake_A = D_A(fake_A.detach())
            loss_D_A_real = criterion_GAN(pred_real_A, real_like(pred_real_A))
            loss_D_A_fake = criterion_GAN(pred_fake_A, fake_like(pred_fake_A))
            loss_D_A = 0.5 * (loss_D_A_real + loss_D_A_fake)
            loss_D_A.backward()
            opt_D_A.step()

            # --------- train D_B ---------
            opt_D_B.zero_grad()
            pred_real_B = D_B(real_B)
            pred_fake_B = D_B(fake_B.detach())
            loss_D_B_real = criterion_GAN(pred_real_B, real_like(pred_real_B))
            loss_D_B_fake = criterion_GAN(pred_fake_B, fake_like(pred_fake_B))
            loss_D_B = 0.5 * (loss_D_B_real + loss_D_B_fake)
            loss_D_B.backward()
            opt_D_B.step()

            if (i + 1) % opt.sample_freq == 0:
                with torch.no_grad():
                    # take first item in batch
                    real_A_vis = denorm(real_A[:1]).cpu()
                    real_B_vis = denorm(real_B[:1]).cpu()
                    fake_B_vis = denorm(fake_B[:1]).cpu()  # G_A2B(real_A)
                    fake_A_vis = denorm(fake_A[:1]).cpu()  # G_B2A(real_B)

                    base = f"e{epoch:03d}_it{i+1:05d}"

                    vutils.save_image(real_A_vis, samples_dir / f"{base}_real_A.png")
                    vutils.save_image(fake_B_vis, samples_dir / f"{base}_fake_B.png")
                    vutils.save_image(real_B_vis, samples_dir / f"{base}_real_B.png")
                    vutils.save_image(fake_A_vis, samples_dir / f"{base}_fake_A.png")

                print(f"[samples] saved real/fake A/B at {base}")


        sched_G.step()
        sched_D_A.step()
        sched_D_B.step()


        with torch.no_grad():
            batch = next(iter(loader))
            A = batch["A"].to(device)[:2]
            B = batch["B"].to(device)[:2]
            fake_B = G_A2B(A)
            fake_A = G_B2A(B)
            grid = torch.cat([denorm(A), denorm(fake_B),
                              denorm(B), denorm(fake_A)], dim=0)
            vutils.save_image(grid, samples_dir / f"epoch_{epoch:03d}.png", nrow=2)
        print("saved samples for epoch", epoch)

        if epoch % opt.save_epoch_freq == 0 or epoch == total_epochs:
            torch.save(G_A2B.state_dict(), ckpt_dir / f"G_A2B_{epoch}.pth")
            torch.save(G_B2A.state_dict(), ckpt_dir / f"G_B2A_{epoch}.pth")
            torch.save(D_A.state_dict(), ckpt_dir / f"D_A_{epoch}.pth")
            torch.save(D_B.state_dict(), ckpt_dir / f"D_B_{epoch}.pth")
            print("saved checkpoints at epoch", epoch)

    print("done.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataroot", type=str, required=True,
                   help="root with trainA/trainB")
    p.add_argument("--out_dir", type=str, default="minimal_output")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.0002)
    p.add_argument("--n_epochs", type=int, default=10)
    p.add_argument("--n_epochs_decay", type=int, default=10)
    p.add_argument("--lambda_cycle", type=float, default=10.0)
    p.add_argument("--lambda_id", type=float, default=0.5)
    p.add_argument("--print_freq", type=int, default=100)
    p.add_argument("--save_epoch_freq", type=int, default=5)
    p.add_argument("--sample_freq", type=int, default=500,   # 👈 add this
                   help="save real/fake samples every N iters")
    args = p.parse_args()

    train(args)
