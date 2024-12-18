import os
import argparse

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}


def write_image_paths(images_path, txt_path):
    # Define valid image extensions

    with open(txt_path, 'w') as f:
        for root, dirs, files in os.walk(images_path):
            for file in files:
                if os.path.splitext(file)[1].lower() in IMAGE_EXTENSIONS:
                    f.write(os.path.join(root, file) + '\n')


parser = argparse.ArgumentParser(description="Write paths of all image files in a folder to a text file.")
parser.add_argument("images_path", type=str, help="Path to the folder containing images.")
parser.add_argument("txt_path", type=str, help="Path to save the text file with image paths.")

args = parser.parse_args()

write_image_paths(args.images_path, args.txt_path)
