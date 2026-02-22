import os
import json
import imageio
from argparse import ArgumentParser
import pandas as pd
import matplotlib.pyplot as plt

from metric_computation.FID import compute_fid
from metric_computation.PR import compute_pr
from metric_computation.CLIP import compute_clip_score

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--generated_images_dir", type=str, required=True, help="Path to the generated images directory")
    parser.add_argument("--prompt_csv_path", type=str, required=True, help="Path to the prompt dataset")
    parser.add_argument("--images_dataset_path", type=str, required=True, help="Path to the images dataset")
    parser.add_argument("--metric_output_dir", type=str, required=True, help="Path to save the metric results JSON")
    parser.add_argument("--debug", action="store_true", help="Enable debug visualizations")
    parser.add_argument("--debug_num_pairs", type=int, default=1, help="Number of image pairs to visualize in debug mode (default: 1)")
    args = parser.parse_args()

    generated_images_dir = args.generated_images_dir
    prompt_csv_path = args.prompt_csv_path
    images_dataset_path = args.images_dataset_path
    metric_output_dir = args.metric_output_dir

    if not os.path.exists(generated_images_dir):
        raise FileNotFoundError(f"Generated images directory {generated_images_dir} does not exist")
    if not os.path.exists(prompt_csv_path):
        raise FileNotFoundError(f"Prompt csv path {prompt_csv_path} does not exist")
    if not os.path.exists(images_dataset_path):
        raise FileNotFoundError(f"Images dataset path {images_dataset_path} does not exist")
    os.makedirs(metric_output_dir, exist_ok=True)

    # read metadata.csv from generated_images_dir
    metadata_path = os.path.join(generated_images_dir, "metadata.csv")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata path {metadata_path} does not exist")
    metadata = pd.read_csv(metadata_path)

    # training dataset metadata
    training_metadata = pd.read_csv(prompt_csv_path)

    # collect prompts directly from the generated images metadata CSV
    prompts = metadata["prompt"].tolist()

    print(f"Number of generated images: {len(metadata)}")
    print(f"Number of prompts: {len(prompts)}")
    print(f"Number of training images: {len(training_metadata)}")

    # for each prompt in the generated images, find the corresponding image in the training dataset
    image_pairs = {
        "generated": [],
        "target": [],
    }
    for _, row in metadata.iterrows():
        # get generated image and prompt
        generated_image, prompt = row["image"], row["prompt"]

        # find the corresponding image in the training dataset
        target_image = training_metadata[training_metadata["prompt"] == prompt]["image"].values[0]
        print(f"Generated image: {generated_image}, Target image: {target_image}")

        if args.debug and len(image_pairs["generated"]) < args.debug_num_pairs:
            image_pairs["generated"].append(imageio.imread(os.path.join(generated_images_dir, generated_image)))
            image_pairs["target"].append(imageio.imread(os.path.join(images_dataset_path, target_image)))

    # TODO: compute aggregated metrics here
    # Image Metrics: FID, Precision-Recall, CLIP
    fid_results = compute_fid(generated_images_dir, images_dataset_path)
    pr_results = compute_pr(generated_images_dir, images_dataset_path)
    clip_results = compute_clip_score(generated_images_dir, images_dataset_path, args={"prompts": prompts})

    print(f"FID: {fid_results['fid']:.4f}  |  mean_dist: {fid_results['mean_dist']:.4f}  |  cov_dist: {fid_results['cov_dist']:.4f}")
    print(f"Precision: {pr_results['precision']:.5f}  |  Recall: {pr_results['recall']:.5f}")
    print(f"CLIP Sim: {clip_results['clipsim']:.4f}")

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
