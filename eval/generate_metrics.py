import os
import imageio
from argparse import ArgumentParser
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--generated_images_dir", type=str, required=True, help="Path to the generated images directory")
    parser.add_argument("--prompt_csv_path", type=str, required=True, help="Path to the prompt dataset")
    parser.add_argument("--images_dataset_path", type=str, required=True, help="Path to the images dataset")
    args = parser.parse_args()

    generated_images_dir = args.generated_images_dir
    prompt_csv_path = args.prompt_csv_path
    images_dataset_path = args.images_dataset_path

    if not os.path.exists(generated_images_dir):
        raise FileNotFoundError(f"Generated images directory {generated_images_dir} does not exist")
    if not os.path.exists(prompt_csv_path):
        raise FileNotFoundError(f"Prompt csv path {prompt_csv_path} does not exist")
    if not os.path.exists(images_dataset_path):
        raise FileNotFoundError(f"Images dataset path {images_dataset_path} does not exist")
    
    # read metadata.csv from generated_images_dir
    metadata_path = os.path.join(generated_images_dir, "metadata.csv")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata path {metadata_path} does not exist")
    metadata = pd.read_csv(metadata_path)
    
    # training dataset metadata
    training_metadata = pd.read_csv(prompt_csv_path)
    
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

        # TODO: compute aggregated metrics here

        # DEBUG: visualize the generated and target images
        generated_image = imageio.imread(os.path.join(generated_images_dir, generated_image))
        target_image = imageio.imread(os.path.join(images_dataset_path, target_image))
        
        image_pairs["generated"].append(generated_image)
        image_pairs["target"].append(target_image)
    
    # DEBUG: visualize the generated and target images in image_pairs in two columns in a grid 
    plt.figure(figsize=(10, 10))
    for i in range(len(image_pairs["generated"])):
        if i == 0:
            plt.suptitle("Generated and Target Images")
        plt.subplot(len(image_pairs["generated"]), 2, 2*i + 1)
        plt.imshow(image_pairs["generated"][i])
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.subplot(len(image_pairs["generated"]), 2, 2*i + 2)
        plt.imshow(image_pairs["target"][i])
        # remove all ticks and grid lines
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
    plt.tight_layout()
    plt.savefig("generated_and_target_images.png")
    plt.close()
    