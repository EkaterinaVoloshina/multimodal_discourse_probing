import argparse
import json
import os
import sys
from io import BytesIO

import requests
from PIL import Image
from tqdm import tqdm


def open_and_save(url, filename=None, dirname=None):
    response = requests.get(url)
    try:
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img.save(os.path.join(dirname, f"{filename}.jpg"))
    except:
        pass


def load_dataset(filename, dirname):
    with open(filename) as f:
        content = json.load(f)

    images = content["images"]

    for num, image in tqdm(enumerate(images)):
        idx = image["id"]
        if f"{idx}.jpg" not in os.listdir(dirname):
            if "url_o" in image:
                img = open_and_save(
                    image["url_o"], filename=image["id"], dirname=dirname
                )
            elif "url_m" in image:
                img = open_and_save(
                    image["url_o"], filename=image["id"], dirname=dirname
                )
            else:
                pass


def main():
    parser = argparse.ArgumentParser(description="Load and save images from VIST.")
    parser.add_argument(
        "-d", "--dirname", type=str, help="The directory with json files", default="dii"
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        help="The output directory with images",
        default="vist_images",
    )

    args = parser.parse_args()

    train_data = "train.description-in-isolation.json"
    val_data = "val.description-in-isolation.json"
    test_data = "test.description-in-isolation.json"

    # print("Loading train...")
    # load_dataset(os.path.join(args.dirname, train_data), args.output_dir)

    # print("Loading validation...")
    # load_dataset(os.path.join(args.dirname, val_data), args.output_dir)

    print("Loading test...")
    load_dataset(os.path.join(args.dirname, test_data), args.output_dir)


if __name__ == "__main__":
    main()
