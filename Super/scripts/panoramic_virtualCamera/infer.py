import os
import argparse
import numpy as np
from PIL import Image
import torch
from config import load_config
from net import build_net

IMAGE_EXT = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')


def normalize(image):
    mn, mx = np.min(image), np.max(image)
    if mx - mn > 1e-8:
        return (image - mn) / (mx - mn)
    return image


def preprocess_single(path, device, size=(256, 455)):
    img = Image.open(path).convert('L')
    if img.size != (size[1], size[0]):
        img = img.resize((size[1], size[0]), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = normalize(arr)
    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)


def preprocess_dual(depth_path, intensity_path, device, h=240, w=1376):
    depth_img = np.array(Image.open(depth_path), dtype=np.float32)
    intensity_img = np.array(Image.open(intensity_path), dtype=np.float32)
    depth_img = normalize(depth_img)[:h, :w]
    intensity_img = normalize(intensity_img)[:h, :w]
    x = torch.tensor(np.stack([intensity_img, depth_img], axis=0), dtype=torch.float32).unsqueeze(0).to(device)
    return x


def save_output(tensor, path):
    arr = tensor.squeeze().cpu().numpy()
    arr = normalize(arr)
    arr = np.clip(arr, 0, 1)
    Image.fromarray((arr * 255).astype(np.uint8)).save(path)


def process_folder_single(model, input_dir, output_dir, device, size):
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(IMAGE_EXT)]
    files.sort()
    if not files:
        return 0
    os.makedirs(output_dir, exist_ok=True)
    n = 0
    for f in files:
        inp = os.path.join(input_dir, f)
        if not os.path.isfile(inp):
            continue
        try:
            x = preprocess_single(inp, device, size)
            with torch.no_grad():
                out = model(x)
            save_output(out, os.path.join(output_dir, os.path.splitext(f)[0] + '.png'))
            n += 1
        except Exception as e:
            print(f"Error {f}: {e}")
    return n


def run_virtual_camera(model_path, base_folder, input_subdir, output_subdir, device, size, use_aspp=True):
    device = torch.device(device)
    if not os.path.isfile(model_path):
        print(f"Model not found: {model_path}")
        return
    from net import SingleChannelNet
    model = SingleChannelNet(use_aspp=use_aspp).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    found = [root for root, _, _ in os.walk(base_folder) if os.path.isdir(os.path.join(root, input_subdir))]
    for root in found:
        n = process_folder_single(model, os.path.join(root, input_subdir), os.path.join(root, output_subdir), device, size)
        if n:
            print(f"Processed {n} images -> {os.path.join(root, output_subdir)}")


def run_panoramic_single(model_path, depth_path, intensity_path, output_dir, device, use_aspp=True):
    device = torch.device(device)
    if not os.path.isfile(model_path):
        print(f"Model not found: {model_path}")
        return
    from net import DualChannelNet
    model = DualChannelNet(use_aspp=use_aspp).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    x = preprocess_dual(depth_path, intensity_path, device)
    with torch.no_grad():
        out, out2 = model(x)
    out = out.cpu().numpy()[0]
    out2 = out2.cpu().numpy()[0]
    save_output(torch.tensor(out[0:1]), os.path.join(output_dir, 'intensity_out1.png'))
    save_output(torch.tensor(out[1:2]), os.path.join(output_dir, 'depth_out1.png'))
    save_output(torch.tensor(out2), os.path.join(output_dir, 'intensity_comp.png'))
    print(f"Saved -> {output_dir}")


def run_panoramic_folder(model_path, base_folder, depth_subdir, intensity_subdir, output_subdir, device, h, w, use_aspp=True):
    device = torch.device(device)
    if not os.path.isfile(model_path):
        print(f"Model not found: {model_path}")
        return
    from net import DualChannelNet
    model = DualChannelNet(use_aspp=use_aspp).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    candidates = []
    for root, _, _ in os.walk(base_folder):
        depth_dir = os.path.join(root, depth_subdir)
        inten_dir = os.path.join(root, intensity_subdir)
        if os.path.isdir(depth_dir) and os.path.isdir(inten_dir):
            candidates.append((root, depth_dir, inten_dir))
    if not candidates:
        print(f"No folders under {base_folder} containing both '{depth_subdir}' and '{intensity_subdir}'")
        return

    for root, depth_dir, inten_dir in candidates:
        out_dir = os.path.join(root, output_subdir)
        os.makedirs(out_dir, exist_ok=True)

        files = [f for f in os.listdir(inten_dir) if f.lower().endswith(IMAGE_EXT)]
        files.sort()
        if not files:
            continue

        n = 0
        for f in files:
            inten_path = os.path.join(inten_dir, f)
            depth_name = f.replace("intensity", "depth")
            depth_path = os.path.join(depth_dir, depth_name)
            if not os.path.isfile(depth_path):
                depth_path = os.path.join(depth_dir, f)
                if not os.path.isfile(depth_path):
                    print(f"[skip] no depth for {inten_path}")
                    continue
            try:
                x = preprocess_dual(depth_path, inten_path, device, h=h, w=w)
                with torch.no_grad():
                    out, out2 = model(x)
                out = out.cpu().numpy()[0]
                out2 = out2.cpu().numpy()[0]
                stem = os.path.splitext(f)[0]
                save_output(torch.tensor(out[0:1]), os.path.join(out_dir, f"{stem}_intensity_out1.png"))
                save_output(torch.tensor(out[1:2]), os.path.join(out_dir, f"{stem}_depth_out1.png"))
                save_output(torch.tensor(out2), os.path.join(out_dir, f"{stem}_intensity_comp.png"))
                n += 1
            except Exception as e:
                print(f"Error {f}: {e}")
        if n:
            print(f"Processed {n} pairs -> {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--view_type', type=str, choices=['virtual_camera', 'panoramic'], required=True)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--base_folder', type=str, default=None)
    parser.add_argument('--depth_path', type=str, default=None)
    parser.add_argument('--intensity_path', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--input_subdir', type=str, default=None)
    parser.add_argument('--output_subdir', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else None
    if cfg:
        if args.model_path is None:
            args.model_path = getattr(getattr(cfg, "infer", cfg), "model_path", None)
        res = cfg.resolution
        size = (res.height, res.width)
        if args.view_type == 'virtual_camera':
            if args.base_folder is None:
                args.base_folder = getattr(cfg.infer, 'base_folder', None)
            if args.input_subdir is None:
                args.input_subdir = getattr(cfg.infer, 'input_subdir', 'depth')
            if args.output_subdir is None:
                args.output_subdir = getattr(cfg.infer, 'output_subdir', 'depth_dense')
        else:
            if args.base_folder is None:
                args.base_folder = getattr(cfg.infer, 'base_folder', None)
    else:
        size = (256, 455)
        if args.view_type == 'virtual_camera':
            args.input_subdir = args.input_subdir or 'depth'
            args.output_subdir = args.output_subdir or 'depth_dense'


    use_aspp = True
    if cfg is not None:
        use_aspp = getattr(cfg, "use_aspp", True)

    if not args.model_path:
        raise SystemExit('model_path is required (either via --model_path or config.infer.model_path)')

    if args.view_type == 'virtual_camera':
        if not args.base_folder:
            raise SystemExit('virtual_camera requires --base_folder')
        run_virtual_camera(args.model_path, args.base_folder, args.input_subdir, args.output_subdir, args.device, size, use_aspp=use_aspp)
    else:
        if args.base_folder:
            if not cfg:
                raise SystemExit('panoramic folder mode requires --config with infer.depth_subdir/intensity_subdir/output_subdir')
            depth_subdir = getattr(cfg.infer, 'depth_subdir', 'depth_view')
            intensity_subdir = getattr(cfg.infer, 'intensity_subdir', 'intensity_view')
            output_subdir = getattr(cfg.infer, 'output_subdir', 'panoramic_out')
            h, w = size
            run_panoramic_folder(args.model_path, args.base_folder, depth_subdir, intensity_subdir, output_subdir, args.device, h, w, use_aspp=use_aspp)
        else:
            if not args.depth_path or not args.intensity_path:
                raise SystemExit('panoramic requires --depth_path and --intensity_path')
            run_panoramic_single(args.model_path, args.depth_path, args.intensity_path, args.output_dir, args.device, use_aspp=use_aspp)
