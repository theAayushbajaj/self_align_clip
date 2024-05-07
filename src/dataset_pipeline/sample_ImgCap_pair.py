#%%
import glob
import random
import shutil
from pathlib import Path
from tqdm import tqdm
import logging
from collections import defaultdict
import json
import argparse
import sys

# Configure logging to output to the console
logging.basicConfig(filename='sample_image.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
sys.stderr = open('sample_image.log', 'a')

args = argparse.ArgumentParser()
args.add_argument("--data_dir", type=str, default=".")
args = args.parse_args()

data_dir = args.data_dir

#%%
# Paths
source_path = Path(data_dir+'/images/')
destination_path = Path(f'{data_dir}/sampled_images_20k/')
destination_path.mkdir(parents=True, exist_ok=True)

# Image extensions to look for
image_extensions = ['*.jpg']

# Collecting images
images = []
for extension in image_extensions:
    images_found = list(source_path.rglob(extension))
    images.extend(images_found)
    logging.info(f'Found {len(images_found)} images with extension {extension}')

# Sampling images
sample_size = min(20_000, len(images))
logging.info(f'Sampling {sample_size} images out of {len(images)}')
sampled_images = random.sample(images, sample_size)
#%%
# Copying images
for image_path in tqdm(sampled_images, desc="Copying images"):
    destination = destination_path / image_path.name
    if not destination.exists():
        shutil.copy(image_path, destination)
    else:
        logging.warning(f'Skipped {image_path} as it already exists in the destination')

logging.info('Completed copying sampled images.')

#%%
def get_blip_captions_by_id_meta(json_data, id_list):
    captions = defaultdict(lambda: '')
    logging.info("Starting to search for captions.")

    for item in tqdm(json_data, desc="Processing items"):
        if item['id'] in id_list:
            assert len(item.get('blip_caption', '')) > 0, f"Caption length for ID {item['id']} is 0."
            captions[item['id']] = item['blip_caption']
            #logging.info(f"Found caption for ID {item['id']}.")

    logging.info("Completed searching for captions.")
    return captions

try:
    with open(f'{data_dir}/blip_laion_cc_sbu_558k_meta.json', 'r') as j:
         blip_caption = json.loads(j.read())

    image_filenames_to_search = [filename.name.split('.')[0] for filename in sampled_images]
    captions = get_blip_captions_by_id_meta(blip_caption, image_filenames_to_search)
except Exception as e:
    logging.error(f"An unexpected error occurred: {e}")

#%%
# Saving captions to a file
captions_file_path = f'{data_dir}/blip_laion_cc_sbu_20k.json'
try:
    with open(captions_file_path, 'w') as file:
        json.dump(captions, file)
    logging.info(f"Captions saved successfully to {captions_file_path}.")
except Exception as e:
    logging.info(f"An error occurred while saving the captions: {e}")

#%%
if __name__ == "__main__":
    pass