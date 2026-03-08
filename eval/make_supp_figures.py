import numpy as np
import sys
import os
import subprocess
from pprint import pprint

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_FIGURES_DIR = "/home/brianchc/foveated_diffusion/paper_figures/supp"


def make_baseline_comparison_figure():
    def make_figure(image_path_set, set_index, num_rows, num_cols, downsample):
        """Call generate_paper_figures.py stack with 18 images, 9 rows x 2 cols."""
        output_path = os.path.join(PAPER_FIGURES_DIR, f"baseline_{set_index + 1}.png")
        os.makedirs(PAPER_FIGURES_DIR, exist_ok=True)
        generate_script = os.path.join(SCRIPT_DIR, "generate_paper_figures.py")
        cmd = [
            sys.executable,
            generate_script,
            "stack",
            "--images",
            *image_path_set,
            "-o",
            output_path,
            "--num-rows",
            str(num_rows),
            "--num-cols",
            str(num_cols),
            "--pad-px",
            "20",
            "--downsample",
            str(downsample),
        ]
        subprocess.run(cmd, check=True, cwd=SCRIPT_DIR)

    def make_image_path_set(image_ids):
        image_path_list = []
        for subset_idx, image_ids in image_ids.items():
            for image_id in image_ids:
                image_path_list.append(f"/home/brianchc/foveated_diffusion/results/eccv2026_final_v2/baseline_comparison_randomized_{subset_idx}/stacked/{image_id:05d}.png")
        np.random.shuffle(image_path_list)
        return image_path_list

    image_ids = {
        1: [2, 6, 7, 13, 52, 55, 62, 86, 28, 30, 31, 32, 39, 95, 45],
        2: [0, 8, 20, 24, 62, 81, 83, 92, 98, 32, 33, 34, 37, 65],
        3: [12, 13, 22, 25, 27, 30, 31, 33, 39, 45, 53, 55, 59, 64, 93, 98, 21, 37, 50, 76, 79, 81, 74, 83, 88]
    }

    hg_image_ids = {
        2: [19, 75],
        3: [16, 86]
    }

    image_path_list = make_image_path_set(image_ids)
    print("total baseline comparison images: ", len(image_path_list))

    # split into sets of 18 images, and make a figure for each set
    for i in range(0, len(image_path_list), 18):
        image_path_set = image_path_list[i:i+18]
        set_index = i // 18
        pprint(image_path_set)
        make_figure(image_path_set, set_index, 9, 2, 4)

    hq_image_path_list = make_image_path_set(hg_image_ids)
    print("total hq baseline comparison images: ", len(hq_image_path_list))
    make_figure(hq_image_path_list, 100, 4, 1, 1)

def make_saliency_figure():
    def make_figure(image_path_set, category):
        try:
            output_path = os.path.join(PAPER_FIGURES_DIR, f"{category}.png")
            os.makedirs(PAPER_FIGURES_DIR, exist_ok=True)
            generate_script = os.path.join(SCRIPT_DIR, "generate_paper_figures.py")
            num_rows = 9
            num_cols = 2
            if category == "saliency_multi":
                num_rows = 4
                num_cols = 2

            cmd = [
                sys.executable,
                generate_script,
                "stack",
                "--images",
                *image_path_set,
                "-o",
                output_path,
                "--num-rows",
                str(num_rows),
                "--num-cols",
                str(num_cols),
                "--pad-px",
                "20",
                "--downsample",
                "4"
            ]
            subprocess.run(cmd, check=True, cwd=SCRIPT_DIR)
        except Exception as e:
            print(f"error making figure for category {category}: {e}")
            return


    image_ids = {
        "saliency_single": [5, 26, 27, 30, 32, 33, 35, 37, 40, 43, 45, 47, 61, 64, 68, 71, 76, 84, 95, 99],
        "saliency_multi": [0, 1, 4, 25, 35, 40, 47, 61, 86, 88],
        "games": [1, 4, 24, 25, 28, 29, 32, 34, 41, 45, 53, 54, 55, 62, 66, 69, 75, 77, 82, 86, 87, 90, 99],
        "robotics": [0, 4, 15, 16, 25, 32, 33, 40, 42, 50, 56, 57, 65, 66, 67, 75, 78, 79, 83, 90],
        "av": [0, 1, 6, 7, 13, 18, 20, 29, 34, 48, 53, 58, 64, 70, 72, 79, 85, 94, 96, 97],
        "bbox_single": [17, 23, 26, 27, 31, 33, 40, 41, 42, 43, 44, 45, 47, 64, 69, 70, 76, 80, 81, 82, 83, 84, 90, 95, 99],
        "bbox_multi": [0, 3, 4, 10, 19, 23, 24, 25, 29, 32, 38, 49, 55, 56, 61, 86, 88, 90, 91, 95, 98, 99],
        "radius": [1, 5, 10, 11, 16, 18, 25, 26, 27, 30, 33, 43, 44, 59, 64, 66, 72, 82],
        "circular": [8, 22, 26, 28, 29, 40, 41, 42, 43, 44, 50, 59, 71,74, 91, 65, 86, 18],
        "multi_circle": [12, 15, 18, 39, 40, 47, 55, 59, 60, 67, 70, 77, 93, 31, 1, 7, 66, 3],
    }
    
    def make_image_path_set(category, image_ids):
        image_path_list = []
        for image_id in image_ids:
            if category == "saliency_single":
                image_path_list.append(f"/home/brianchc/foveated_diffusion/results/eccv2026_final_v2/saliency_generation/stacked/{image_id:03d}.png")
            elif category == "saliency_multi":
                image_path_list.append(f"/home/brianchc/foveated_diffusion/results/eccv2026_final_v2/saliency_generation_multi_object/stacked/{image_id:03d}.png")
            elif category == "bbox_single":
                image_path_list.append(f"/home/brianchc/foveated_diffusion/results/eccv2026_final_v2/bbox_generation/stacked/{image_id:03d}.png")
            elif category == "bbox_multi":
                image_path_list.append(f"/home/brianchc/foveated_diffusion/results/eccv2026_final_v2/bbox_multi_object_generation/stacked/{image_id:03d}.png")
            elif category in ["games", "robotics", "av"]:
                image_path_list.append(f"/home/brianchc/foveated_diffusion/results/eccv2026_final_v2/supp/{category}_claude/stacked/{image_id:03d}.png")
            elif category in ["radius", "circular", "multi_circle"]:
                image_path_list.append(f"/home/brianchc/foveated_diffusion/results/eccv2026_final_v2/supp/foveation_trajectory/ours_{category}/stacked/{image_id:03d}.png")
            else:
                raise ValueError(f"Invalid category: {category}")
        np.random.shuffle(image_path_list)
        return image_path_list

    for category, image_ids in image_ids.items():
        image_path_list = make_image_path_set(category, image_ids)
        print(f"total images for category {category}: ", len(image_path_list))
        num_images = 18
        if category == "saliency_multi":
            num_images = 8
        for i in range(0, len(image_path_list), num_images):
            image_path_set = image_path_list[i:i+num_images]
            set_index = i // num_images
            make_figure(image_path_set, category)

if __name__ == "__main__":
    #make_baseline_comparison_figure()
    make_saliency_figure()