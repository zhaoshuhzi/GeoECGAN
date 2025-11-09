import torch, argparse
from geoecgan.models import GeometricEigenmodes, GeneratorGeoECGAN, Discriminator
from geoecgan.data import HBN_EEG_WindowDataset
import torch.nn.functional as F

def hinge_d_loss(D_real, D_fake): return F.relu(1.-D_real).mean() + F.relu(1.+D_fake).mean()
def hinge_g_loss(D_fake): return -D_fake.mean()
def cosine_align(a,b): return 1.0 - F.cosine_similarity(a,b,dim=-1).mean()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--regions", type=int, default=360)
    p.add_argument("--modes", type=int, default=100)
    p.add_argument("--latent", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    R = args.regions
    A = torch.rand(R, R); A = (A + A.T)/2; A.fill_diagonal_(0.0)
    gem = GeometricEigenmodes(n_modes=args.modes)
    modes = gem(A=A).to(args.device)

    G = GeneratorGeoECGAN(n_regions=R, n_modes=args.modes, latent_dim=args.latent).to(args.device)
    D = Discriminator(latent_dim=args.latent).to(args.device)

    opt_g = torch.optim.AdamW(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.AdamW(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    dl = torch.utils.data.DataLoader(HBN_EEG_WindowDataset(n_samples=32, n_regions=R),
                                     batch_size=args.batch_size, shuffle=True, drop_last=True)
    for step, batch in enumerate(dl, 1):
        batch = batch.to(args.device)
        out = G(batch, modes)
        z = out["z"]; z_real = torch.randn_like(z)

        opt_d.zero_grad()
        d_real = D(z_real.detach()); d_fake = D(z.detach())
        loss_d = hinge_d_loss(d_real, d_fake); loss_d.backward(); opt_d.step()

        opt_g.zero_grad()
        d_fake = D(z)
        loss_g = hinge_g_loss(d_fake) + 0.1*cosine_align(out["z_npi"], out["z_geo"]) \
                 + 0.1*F.mse_loss(out["recon"], batch[:, -1, :])
        loss_g.backward(); opt_g.step()

        if step % 2 == 0:
            print(f"step {step:03d}  loss_d={loss_d.item():.4f}  loss_g={loss_g.item():.4f}")
    print("done.")

if __name__ == "__main__":
    main()
