import os
import sys
import argparse
from datetime import datetime
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from config import load_config
from dataset import build_dataset, psnr
from net import build_net

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


DEFAULT_CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ckpt')


class TeeLogger:
    """Write terminal output to both stream and a shared log file."""
    def __init__(self, log_file, stream):
        self.terminal = stream
        self.log = log_file
        self._is_stdout = stream is sys.stdout
    def write(self, message):
        self.terminal.write(message)
        if self.log:
            self.log.write(message)
            self.log.flush()
    def flush(self):
        self.terminal.flush()
        if self.log:
            self.log.flush()
    def close(self):
        if self._is_stdout:
            sys.stdout = self.terminal
        else:
            sys.stderr = self.terminal


def compute_rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()


def compute_mae(pred, target):
    return torch.mean(torch.abs(pred - target)).item()


def save_batch_grayscale(output_tensor, filename_prefix, epoch, save_dir):
    save_dir = os.path.join(save_dir, f"epoch_{epoch}")
    os.makedirs(save_dir, exist_ok=True)
    for i in range(output_tensor.size(0)):
        arr = output_tensor[i, 0].detach().cpu().numpy()
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
        Image.fromarray(arr.astype(np.uint8)).save(os.path.join(save_dir, filename_prefix[i]))


def save_model(model, epoch, save_dir, title):
    epoch_dir = os.path.join(save_dir, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)
    path = os.path.join(epoch_dir, f"{title}_epoch_{epoch}.pth")
    torch.save(model.state_dict(), path)


def train_virtual_camera(model, train_loader, test_loader, epochs, cfg):
    loss_fn = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=cfg.train.lr)
    model = model.to(device)
    best_psnr = -1.0
    save_dir = cfg.train.save_dir
    raw_save_dir = cfg.train.raw_save_dir or os.path.join(save_dir, 'raw')
    save_every = getattr(cfg.train, 'save_every', 4)
    mode_label = getattr(cfg.data, 'mode', 'depth')

    for i in range(epochs):
        model.train()
        torch.cuda.empty_cache()
        epoch_loss = 0
        ite = 0
        pbar = tqdm(train_loader, desc=f"Train (virtual_camera) epoch {i+1}/{epochs}", ncols=100)
        for step, (x, y, name) in enumerate(pbar):
            if step % 10 == 0 and torch.cuda.is_available():
                print(f'Step {step}, GPU: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB')
            opt.zero_grad()
            x, y = x.to(device), y.to(device)
            try:
                out = model(x)
                l = loss_fn(out, y)
                l.backward()
                opt.step()
                epoch_loss += l.item()
                ite += 1
                if step % 50 == 0:
                    torch.cuda.empty_cache()
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    torch.cuda.empty_cache()
                    continue
                raise e

        epoch_loss /= max(ite, 1)
        print(f'epoch: {i} | loss: {epoch_loss:.8f}')
        wandb.log({"epoch": i, "train_loss": epoch_loss})

        model.eval()
        torch.cuda.empty_cache()
        test_psnr_val = test_rmse_val = test_mae_val = 0
        test_i = 0
        do_save = (i + 1) % save_every == 0 or i == epochs - 1
        with torch.no_grad():
            for idx, (x, y, name) in enumerate(test_loader):
                x, y = x.to(device), y.to(device)
                try:
                    out = model(x)
                    for j in range(len(name)):
                        pred = out[j, 0].cpu().detach()
                        target = y[j, 0].cpu().detach()
                        test_psnr_val += psnr(pred.numpy(), target.numpy())
                        test_rmse_val += compute_rmse(pred, target)
                        test_mae_val += compute_mae(pred, target)
                        test_i += 1
                    if do_save:
                        save_batch_grayscale(out, name, i, raw_save_dir)
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        torch.cuda.empty_cache()
                        continue
                    raise e
        if test_i == 0:
            continue
        avg_psnr = test_psnr_val / test_i
        avg_rmse = test_rmse_val / test_i
        avg_mae = test_mae_val / test_i
        print(f'Avg PSNR: {avg_psnr:.2f} | RMSE: {avg_rmse:.4f} | MAE: {avg_mae:.4f}')
        wandb.log({"epoch": i, "test_psnr": avg_psnr, "test_rmse": avg_rmse, "test_mae": avg_mae})
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            save_model(model, 'best', save_dir, f'range_panoramic_{mode_label}_best')
        if do_save:
            save_model(model, i, save_dir, f'range_panoramic_{mode_label}')
    print(f'Best PSNR: {best_psnr:.2f}')


def train_panoramic(model, train_loader, test_loader, epochs, cfg):
    loss_fn = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=cfg.train.lr)
    model = model.to(device)
    save_dir = cfg.train.save_dir
    pred_save_dir = os.path.join(save_dir, 'pred')
    com_save_dir = os.path.join(save_dir, 'com')

    for i in range(epochs):
        model.train()
        torch.cuda.empty_cache()
        epoch_loss = 0
        ite = 0
        pbar = tqdm(train_loader, desc=f"Train (panoramic) epoch {i+1}/{epochs}", ncols=100)
        for step, (x, y, com, name) in enumerate(pbar):
            opt.zero_grad()
            x, y = x.to(device), y.to(device)
            com = com.to(device).unsqueeze(1)
            out, out2 = model(x)
            l = loss_fn(out, y) + loss_fn(out2, com)
            l.backward()
            opt.step()
            epoch_loss += l.item()
            ite += 1
        epoch_loss /= max(ite, 1)
        print(f'epoch: {i} | loss: {epoch_loss:.8f}')
        wandb.log({"epoch": i, "train_loss": epoch_loss})
        save_model(model, i, save_dir, 'range_panoramic')

        model.eval()
        test_psnr_r = test_psnr_d = test_rmse = test_mae = 0
        test_i = 0
        for idx, (x, y, com, name) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                out, out2 = model(x)
            for j in range(len(name)):
                pred = out[j, 0].cpu().detach()
                target = y[j, 0].cpu().detach()
                test_psnr_r += psnr(out[j, 0].cpu().numpy(), y[j, 0].cpu().numpy())
                test_psnr_d += psnr(out[j, 1].cpu().numpy(), y[j, 1].cpu().numpy())
                test_rmse += compute_rmse(pred, target)
                test_mae += compute_mae(pred, target)
                test_i += 1
            save_batch_grayscale(out, name, i, pred_save_dir)
            save_batch_grayscale(out2, name, i, com_save_dir)
        if test_i == 0:
            continue
        print(f'Avg PSNR (R): {test_psnr_r/test_i:.2f} | PSNR (D): {test_psnr_d/test_i:.2f} | RMSE: {test_rmse/test_i:.4f} | MAE: {test_mae/test_i:.4f}')
        wandb.log({"epoch": i, "test_psnr_reflectance": test_psnr_r/test_i, "test_psnr_depth": test_psnr_d/test_i, "test_rmse": test_rmse/test_i, "test_mae": test_mae/test_i})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--view_type', type=str, choices=['virtual_camera', 'panoramic'], default=None)
    parser.add_argument('--mode', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.view_type:
        cfg.view_type = args.view_type
    if args.data_root:
        cfg.data.data_root = args.data_root
    if args.mode:
        cfg.data.mode = args.mode
    if args.save_dir:
        cfg.train.save_dir = args.save_dir
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.epochs is not None:
        cfg.train.epochs = args.epochs

    base_ckpt = cfg.train.save_dir or DEFAULT_CKPT_DIR
    base_ckpt = os.path.normpath(os.path.abspath(base_ckpt))
    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{cfg.view_type}"
    run_dir = os.path.join(base_ckpt, run_name)
    os.makedirs(run_dir, exist_ok=True)
    cfg.train.save_dir = run_dir
    cfg.train.raw_save_dir = os.path.join(run_dir, 'samples')
    os.environ["WANDB_DIR"] = run_dir

    log_path = os.path.join(run_dir, 'train.log')
    log_file = open(log_path, 'w', encoding='utf-8')
    tee_stdout = TeeLogger(log_file, sys.stdout)
    tee_stderr = TeeLogger(log_file, sys.stderr)
    sys.stdout = tee_stdout
    sys.stderr = tee_stderr
    print(f"Run dir: {run_dir}")
    print(f"Log file: {log_path}")

    if cfg.view_type == 'virtual_camera' and not (cfg.data.data_root or (cfg.data.coursedic and cfg.data.finedic)):
        tee_stdout.close()
        tee_stderr.close()
        raise SystemExit('Provide data_root or coursedic+finedic in config / --data_root')

    wandb.init(
        project=getattr(cfg.wandb, 'project', 'range_panoramic'),
        config=vars(cfg),
        dir=run_dir,
    )

    traindataset = build_dataset(cfg, 'train')
    testdataset = build_dataset(cfg, 'test')
    trainloader = DataLoader(
        traindataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        drop_last=(cfg.view_type == 'virtual_camera'),
        num_workers=getattr(cfg.train, 'num_workers', 2),
        pin_memory=getattr(cfg.train, 'pin_memory', False)
    )
    testloader = DataLoader(testdataset, batch_size=1, shuffle=False, drop_last=False, num_workers=1)

    use_aspp = getattr(cfg, "use_aspp", True)
    model = build_net(cfg.view_type, use_aspp=use_aspp)
    try:
        if cfg.view_type == 'virtual_camera':
            train_virtual_camera(model, trainloader, testloader, cfg.train.epochs, cfg)
        else:
            train_panoramic(model, trainloader, testloader, cfg.train.epochs, cfg)
    finally:
        tee_stdout.close()
        tee_stderr.close()
        log_file.close()
    wandb.finish()
