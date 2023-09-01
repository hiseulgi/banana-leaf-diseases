from helper import LitMobileNet, download_image, predict_image

import torch
from torchvision import transforms
import wandb

import argparse
import os
from pathlib import Path
from PIL import Image
import time
import warnings
warnings.filterwarnings("ignore")

global_start_time = time.time()

# check model checkpoint
artifact_dir = "artifacts/model-7gtc518c:v49"

if os.path.exists(artifact_dir):
    print(f"The artifact checkpoint '{artifact_dir}' exists.")
else:
    print(f"The artifact checkpoint '{artifact_dir}' does not exists.")
    print(f"Downloading artifcat checkpoint from wandb.")

    checkpoint_reference = "hiseulgi/banana-leaf-diseases/model-7gtc518c:v49"

    run = wandb.init(project="banana-leaf-diseases")
    artifact = run.use_artifact(checkpoint_reference, type="model")
    artifact_dir = artifact.download()
    wandb.finish()

# new model from checkpoint
class_names = ['cordana', 'healthy', 'pestalotiopsis', 'sigatoka']

device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    model = LitMobileNet.load_from_checkpoint(
        Path(artifact_dir) / "model.ckpt")
else:
    model = LitMobileNet.load_from_checkpoint(Path(
        artifact_dir) / "model.ckpt", map_location=torch.device("cpu"), encoder_map_location=torch.device("cpu"))
print(f"Model run on {device}")


# main function
def main():
    parser = argparse.ArgumentParser(
        description="Banana Leaf Diseases Classification")
    parser.add_argument("--input", required=True,
                        help="Path to a directory or URL containing an image.")
    args = parser.parse_args()
    input_path = args.input

    # check directory or url
    if os.path.exists(input_path):
        print(f"'{input_path}' is a valid directory path.")
        image_input = Image.open(input_path)

    elif input_path.startswith("http://") or input_path.startswith("https://"):
        print(f"'{input_path}' is a valid URL.")
        download_dir = download_image(input_path)
        image_input = Image.open(download_dir)

    else:
        print(f"'{input_path}' is neither a valid directory nor a valid URL.")
        return

    # transform image to model requirements
    predict_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    transformed_image = predict_transform(image_input)

    print("Predicting image!")
    inference_start_time = time.time()

    probability = predict_image(model, transformed_image, device)
    prediction = class_names[torch.argmax(probability)]

    print()
    print("===========================")
    print("Prediction Result")
    print("===========================")
    for i, label in enumerate(class_names):
        formatted_prob = "{:.4f}".format(probability[i])
        print(f"{i}. {label} - {formatted_prob}")
    print()
    print(f"Final prediction: {prediction}")
    print("===========================")

    end_time = time.time()
    global_runtime = end_time - global_start_time
    inference_runtime = end_time - inference_start_time
    print(f"Global runtime: {global_runtime:.2f} seconds")
    print(f"Inference runtime: {inference_runtime:.2f} seconds")


# main loop
if __name__ == "__main__":
    main()
