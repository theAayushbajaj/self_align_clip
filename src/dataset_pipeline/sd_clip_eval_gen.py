from tqdm import tqdm
import gc
import torch
from PIL import Image
import json
import logging
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
#from diffusers.models.attention_processor import AttnProcessor2_0
from torchmetrics.functional.multimodal import clip_score
from functools import partial
import pandas as pd
import numpy as np
import argparse
import sys
import os
torch.cuda.empty_cache()

# set numpy and torch seed
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

# Configure logging
logging.basicConfig(filename='clip_eval.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
sys.stderr = open('clip_eval.log', 'a')

def get_json_data(path):
    with open(path, 'r') as j:
        img_cap_pair = json.loads(j.read())
    
    logging.info("Captions loaded")
    return img_cap_pair


def load_image(image_path):
    target_size = (512, 512)
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        img_resized = img.resize(target_size)
        return np.array(img_resized)

def generate_images(batch_captions, pipeline, num_images_per_prompt=1):
    generated_images = pipeline(prompt=batch_captions, num_images_per_prompt=num_images_per_prompt, output_type="np").images
    return generated_images
    
def calculate_clip_score(images, prompts, clip_score_fn):
    images_int = (images * 255).astype("uint8")
    clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
    return round(float(clip_score), 4)

def save_images(images, filenames, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for image, filename in zip(images, filenames):
        # Convert the image to uint8 data type
        image = (image * 255).astype(np.uint8)
        # If the image has a single-channel (grayscale), convert it to 3 channels
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        # Save the image
        image_path = os.path.join(save_dir, f"{filename}.jpg")
        Image.fromarray(image).save(image_path)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--data_dir", type=str, default=".")
    args = args.parse_args()

    data_dir = args.data_dir

    model_id = "stabilityai/stable-diffusion-2-base"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scheduler = EulerDiscreteScheduler.from_pretrained(model_id,subfolder="scheduler",
                                                       #local_files_only=True # Uncomment this line if you have downloaded the scheduler files
                                                       )
    
    sd_pipeline = StableDiffusionPipeline.from_pretrained(model_id, 
                                                          scheduler=scheduler, 
                                                          torch_dtype=torch.float16, 
                                                          use_safetensors=True,
                                                          #local_files_only=True # Uncomment this line if you have downloaded the diffusion model files
                                                          ).to(device)

    logging.info("Stable Diffusion pipeline loaded")

    output_dir = f"{data_dir}/generated_images_20k"

    img_cap_pair = get_json_data(f"{data_dir}/blip_laion_cc_sbu_20k.json")
    filenames = list(img_cap_pair.keys())
    captions = list(img_cap_pair.values())

    batch_size = 32
    clip_score_fn = partial(clip_score, 
                            model_name_or_path="openai/clip-vit-base-patch16")
    results = []
    original_image_dir = data_dir+"/sampled_images_20k/"

    total_batches = len(img_cap_pair) // batch_size + (1 if len(img_cap_pair) % batch_size > 0 else 0)
    with torch.no_grad():
        for i in tqdm(range(total_batches), desc="Processing batches"):
            # Extract the current batch of captions
            batch_captions = captions[i*batch_size:(i+1)*batch_size]
            batch_filenames = filenames[i*batch_size:(i+1)*batch_size]
            
            batch_generated_images = generate_images(batch_captions=batch_captions, pipeline=sd_pipeline, num_images_per_prompt=1)
            save_images(batch_generated_images, batch_filenames, output_dir)
            batch_original_images = np.array([load_image(original_image_dir + filename + '.jpg') for filename in batch_filenames])
            
            batch_generated_clip_score = calculate_clip_score(images=batch_generated_images, prompts=batch_captions, clip_score_fn=clip_score_fn)
            batch_original_clip_score = calculate_clip_score(images=batch_original_images, prompts=batch_captions, clip_score_fn=clip_score_fn)
            
            del batch_generated_images, batch_original_images
            torch.cuda.empty_cache() 
            gc.collect() 
            results.append({
                "Batch": i+1,
                "Generated Images CLIP Score": batch_generated_clip_score,
                "Original Images CLIP Score": batch_original_clip_score
            })


    df_results = pd.DataFrame(results)
    df_results.to_csv(f"../batch_clip_scores_comparison.csv", index=False)
    torch.cuda.empty_cache()

