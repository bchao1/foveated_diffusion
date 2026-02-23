import os
import sys
import json
import numpy as np
import imageio
import torch
import torch.nn.functional as F
import torchvision
from argparse import ArgumentParser
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

def crop_and_resize(image, target_height, target_width):
    width, height = image.size
    scale = max(target_width / width, target_height / height)
    image = torchvision.transforms.functional.resize(
        image,
        (round(height * scale), round(width * scale)),
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
    )
    image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
    return image


def make_per_image_foveation_transform(mask_paths, device, vmin, vmax):
    """Return a stateful batch transform that applies per-image foveation masks.

    mask_paths : list of str-or-None, aligned with the image_paths list passed
                 to the metric extraction function.
    device     : torch device the batch tensor lives on.
    vmin/vmax  : clamp range matching the batch tensor value range
                 (inception uses -1/1, CLIP uses 0/1).

    Each call processes the next len(batch) masks in order, then advances
    the internal counter by that amount.
    """
    state = {'idx': 0}

    def transform(batch):
        B = batch.shape[0]
        imgs = list(batch.unbind(0))
        for i in range(B):
            mask_path = mask_paths[state['idx'] + i]
            if mask_path is not None:
                img = imgs[i]                               # (3, H, W)
                H, W = img.shape[1], img.shape[2]
                mask_arr = np.array(
                    Image.open(mask_path).convert('L'), dtype=np.float32
                ) / 255.0
                mask_t = torch.from_numpy(mask_arr).to(batch.device)[None, None]
                mask_hw = F.interpolate(
                    mask_t, size=(H, W), mode='bilinear', align_corners=False
                )
                img4 = img.unsqueeze(0)
                low_res = F.interpolate(
                    img4, size=(H // 2, W // 2),
                    mode='bicubic', align_corners=False, antialias=True,
                ).clamp(vmin, vmax)
                low_res_up = F.interpolate(
                    low_res, size=(H, W),
                    mode='bicubic', align_corners=False, antialias=True,
                ).clamp(vmin, vmax)
                imgs[i] = (low_res_up * (1 - mask_hw) + img4 * mask_hw).squeeze(0)
        state['idx'] += B
        return torch.stack(imgs)

    return transform


from metric_computation.FID import compute_fid
from metric_computation.PR import compute_pr
from metric_computation.CLIP import compute_clip_score, _load_openclip
from metric_computation.inception_utils import load_inception

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--generated_images_dir", type=str, default=None, help="Path to the generated images directory")
    parser.add_argument("--images_dataset_path", type=str, default=None, help="Path to the images dataset (must contain metadata.jsonl)")
    parser.add_argument("--metric_output_dir", type=str, default=None, help="Path to save the metric results JSON")
    parser.add_argument("--device", type=str, default=None, help="PyTorch device string (e.g. 'cuda:0', 'cpu'). Defaults to cuda:0 if available, else cpu.")
    parser.add_argument("--foveate_real", action="store_true", help="Apply per-image foveation mask (mask_*.png) to real images before computing metrics")
    parser.add_argument("--downsample_real", action="store_true", help="Bicubic 2x downsample real images before computing metrics")
    parser.add_argument("--debug", action="store_true", help="Enable debug visualizations")
    parser.add_argument("--debug_num_pairs", type=int, default=1, help="Number of image pairs to visualize in debug mode (default: 1)")
    parser.add_argument("--test", action="store_true", help="Smoke-test: load models, run on dummy data, check shapes, then exit")
    args = parser.parse_args()

    device = args.device if args.device is not None else ('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args.test:
        B = 4

        # --- Inception (StyleGAN) ---
        print("[TEST] Loading Inception model ...")
        inception_net, inception_style = load_inception(
            {'type': 'StyleGAN'}, metric='FID', map_location='cpu')
        inception_net = inception_net.to(device).eval()

        dummy_imgs = torch.randint(0, 256, (B, 3, 256, 256), dtype=torch.uint8).to(device)
        with torch.no_grad():
            feat = inception_net(dummy_imgs, return_features=True)
            pred = F.linear(feat, inception_net.output.weight).softmax(dim=1)

        assert feat.shape == (B, 2048), f"Inception features: expected ({B}, 2048), got {feat.shape}"
        assert pred.shape == (B, 1008), f"Inception preds:    expected ({B}, 1008), got {pred.shape}"
        print(f"[TEST] Inception features: {tuple(feat.shape)}  OK")
        print(f"[TEST] Inception preds:    {tuple(pred.shape)}  OK")

        # --- CLIP ---
        print("[TEST] Loading CLIP model ...")
        clip_model, tokenizer = _load_openclip(device=device)
        image_size = clip_model.visual.image_size
        if isinstance(image_size, tuple):
            image_size = image_size[0]
        image_size = int(image_size)

        dummy_imgs_clip = torch.rand(B, 3, image_size, image_size, dtype=torch.float16).to(device)
        dummy_text = tokenizer(["a test image"] * B).to(device)
        with torch.no_grad():
            out = clip_model(dummy_imgs_clip, dummy_text)
            img_feat = out['image_features']
            txt_feat = out['text_features']

        assert img_feat.shape[0] == B, f"CLIP image features: expected batch {B}, got {img_feat.shape[0]}"
        assert txt_feat.shape[0] == B, f"CLIP text features:  expected batch {B}, got {txt_feat.shape[0]}"
        assert img_feat.shape == txt_feat.shape, \
            f"CLIP feature shapes mismatch: {img_feat.shape} vs {txt_feat.shape}"
        print(f"[TEST] CLIP image features: {tuple(img_feat.shape)}  OK")
        print(f"[TEST] CLIP text features:  {tuple(txt_feat.shape)}  OK")

        print("[TEST] All shape checks passed.")
        sys.exit(0)

    # Validate required path args (not needed for --test)
    if args.generated_images_dir is None:
        parser.error("--generated_images_dir is required")
    if args.images_dataset_path is None:
        parser.error("--images_dataset_path is required")
    if args.metric_output_dir is None:
        parser.error("--metric_output_dir is required")

    generated_images_dir = args.generated_images_dir
    images_dataset_path = args.images_dataset_path
    metric_output_dir = args.metric_output_dir

    if not os.path.exists(generated_images_dir):
        raise FileNotFoundError(f"Generated images directory {generated_images_dir} does not exist")
    if not os.path.exists(images_dataset_path):
        raise FileNotFoundError(f"Images dataset path {images_dataset_path} does not exist")
    os.makedirs(metric_output_dir, exist_ok=True)

    # Load prompts and image names from prompt_list.txt.
    # Each valid line has the format: "<image_name>: <prompt>"
    # The corresponding image file is img_<image_name>.png in generated_images_dir.
    # Lines that don't contain ": " are skipped.
    prompt_list_path = os.path.join(generated_images_dir, "prompt_list.txt")
    if not os.path.exists(prompt_list_path):
        raise FileNotFoundError(f"prompt_list.txt not found: {prompt_list_path}")

    gen_image_paths = []
    prompts = []
    with open(prompt_list_path, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            if ': ' not in line:
                continue  # skip invalid lines
            image_name, prompt = line.split(': ', 1)
            gen_image_path = os.path.join(generated_images_dir, f'img_{image_name.strip()}.png')
            if not os.path.exists(gen_image_path):
                print(f"WARNING: Generated image not found: {gen_image_path}. Skipping.")
                continue
            gen_image_paths.append(gen_image_path)
            prompts.append(prompt)

    # Training dataset metadata from images_dataset_path/metadata.jsonl (keys: "file_name", "text").
    training_metadata_path = os.path.join(images_dataset_path, "metadata.jsonl")
    if not os.path.exists(training_metadata_path):
        raise FileNotFoundError(f"Training metadata not found: {training_metadata_path}")
    training_metadata = pd.read_json(training_metadata_path, lines=True)

    # Normalize training metadata text: strip surrounding double-quotes and unescape \".
    # In metadata.jsonl the text field is stored as: "A photo of \"Alice\""
    # while prompt_list.txt stores the raw prompt: A photo of "Alice"
    def _norm_jsonl_prompt(s):
        s = s.strip()
        if s.startswith('"') and s.endswith('"'):
            s = s[1:-1]
        return s.replace('\\"', '"')

    training_metadata["_text_norm"] = training_metadata["text"].apply(_norm_jsonl_prompt)

    print(f"Number of generated images: {len(gen_image_paths)}")
    print(f"Number of prompts: {len(prompts)}")
    print(f"Number of training images: {len(training_metadata)}")

    # For each generated image, find the corresponding real image via prompt matching.
    real_image_paths = []
    image_pairs = {"generated": [], "target": []}
    for gen_path, prompt in zip(gen_image_paths, prompts):
        target_image = training_metadata[training_metadata["_text_norm"] == prompt]["file_name"].values[0]
        print(f"Generated image: {os.path.basename(gen_path)}, Target image: {target_image}")
        real_image_paths.append(os.path.join(images_dataset_path, target_image))
        if args.debug and len(image_pairs["generated"]) < args.debug_num_pairs:
            image_pairs["generated"].append(imageio.imread(gen_path))
            image_pairs["target"].append(imageio.imread(os.path.join(images_dataset_path, target_image)))

    # Derive per-image foveation mask paths from generated image filenames.
    # img_XXXXXXXX.png  ->  mask_XXXXXXXX.png  (same directory).
    foveation_mask_paths = []
    for gp in gen_image_paths:
        basename = os.path.basename(gp)
        mask_basename = 'mask_' + basename[4:] if basename.startswith('img_') else 'mask_' + basename
        mask_path = os.path.join(os.path.dirname(gp), mask_basename)
        if not os.path.exists(mask_path):
            print(f"ERROR: Foveation mask not found for {basename}: {mask_path}. "
                  "Treating foveate_real=False for this image.")
            foveation_mask_paths.append(None)
        else:
            foveation_mask_paths.append(mask_path)

    # Preprocessing transform: crop and resize each real image to 1024x1024
    # before foveation/downsampling and inception feature extraction.
    pil_transform = lambda img: crop_and_resize(img, 1024, 1024)

    # Image size check: sample first matched pair to warn about resolution mismatches.
    # When downsample_real=True, generated images are expected to be at half the
    # resolution of the real images (H/2 x W/2).
    if gen_image_paths and real_image_paths:
        _gs = Image.open(gen_image_paths[0]).size   # PIL size is (W, H)
        _rs = pil_transform(Image.open(real_image_paths[0])).size  # size after preprocessing
        if args.downsample_real:
            expected = (_rs[0] // 2, _rs[1] // 2)
            if _gs != expected:
                print(f"WARNING: downsample_real=True but generated image size {_gs} "
                      f"!= expected half-resolution {expected}.")
        else:
            if _gs != _rs:
                print(f"WARNING: generated image size {_gs} != real image size {_rs}. "
                      "Metrics may be invalid.")

    # cache generated image Inception features so FID and PR share one extraction pass
    gen_feats_cache = os.path.join(metric_output_dir, "gen_feats_cache.pkl")

    # Each metric gets a fresh transform instance so the per-image counter starts at 0.
    def _make_fov_transform(vmin, vmax):
        if args.foveate_real:
            return make_per_image_foveation_transform(foveation_mask_paths, device, vmin, vmax)
        return None

    # TODO: compute aggregated metrics here
    fid_results  = compute_fid(gen_image_paths, real_image_paths,
                                args={
                                    "generation_feats_pkl": gen_feats_cache,
                                    "reference_pkl": None,  # always use real_image_paths, not ImageNet PKL
                                    "resize": False,        # real images are already 1024x1024 after pil_transform
                                },
                                foveate_real=args.foveate_real,
                                downsample_real=args.downsample_real,
                                real_pil_transform=pil_transform,
                                real_image_transform=_make_fov_transform(-1, 1),
                                device=device)
    pr_results   = compute_pr(gen_image_paths, real_image_paths,
                               args={
                                   "generation_feats_pkl": gen_feats_cache,
                                   "resize": False,         # real images are already 1024x1024 after pil_transform
                               },
                               foveate_real=args.foveate_real,
                               downsample_real=args.downsample_real,
                               real_pil_transform=pil_transform,
                               real_image_transform=_make_fov_transform(-1, 1),
                               device=device)
    clip_results = compute_clip_score(gen_image_paths,
                                       args={"prompts": prompts},
                                       foveate_real=args.foveate_real,
                                       downsample_real=args.downsample_real,
                                       device=device,
                                       real_paths=real_image_paths,
                                       real_pil_transform=pil_transform,
                                       real_image_transform=_make_fov_transform(0, 1))

    print(f"FID: {fid_results['fid']:.4f}  |  mean_dist: {fid_results['mean_dist']:.4f}  |  cov_dist: {fid_results['cov_dist']:.4f}")
    print(f"Precision: {pr_results['precision']:.5f}  |  Recall: {pr_results['recall']:.5f}")
    print(f"CLIP Sim (gen): {clip_results['clipsim_gen']:.4f}  |  CLIP Sim (real): {clip_results['clipsim_real']:.4f}")

    all_metrics = {**fid_results, **pr_results, **clip_results}
    metrics_output_path = os.path.join(metric_output_dir, "metrics.json")
    with open(metrics_output_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Metrics saved to {metrics_output_path}")

    if args.debug:
        n_pairs = len(image_pairs["generated"])
        plt.figure(figsize=(10, 5 * n_pairs))
        plt.suptitle("Generated and Target Images")
        for i in range(n_pairs):
            plt.subplot(n_pairs, 2, 2*i + 1)
            plt.imshow(image_pairs["generated"][i])
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.subplot(n_pairs, 2, 2*i + 2)
            plt.imshow(image_pairs["target"][i])
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
        plt.tight_layout()
        plt.savefig("generated_and_target_images.png")
        plt.close()
