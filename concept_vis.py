from diffusers import StableDiffusionPipeline
from diffusers.schedulers import LMSDiscreteScheduler
from random import randrange
import torch
from diffusers import StableDiffusionPipeline
from diffusers.schedulers import LMSDiscreteScheduler, DDPMScheduler, DPMSolverMultistepScheduler, PNDMScheduler
import timm
import torchvision.transforms as transforms
import requests
import torch.optim as optim
import numpy as np
from PIL import Image, ImageDraw
import argparse
import os
import glob
from pathlib import Path
import math

# initialize stable diffusion pipeline
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base")
pipe.to("cuda")
scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.scheduler = scheduler
orig_embeddings = pipe.text_encoder.text_model.embeddings.token_embedding.weight.clone().detach()
pipe.text_encoder.text_model.embeddings.token_embedding.weight.requires_grad_(False)

concept = 'dog'
folder = f'./{concept}'
# load coefficients
alphas_dict = torch.load(f'{folder}/best_alphas.pt').detach_().requires_grad_(False)
# load vocabulary
dictionary = torch.load(f'{folder}/dictionary.pt')

sorted_alphas, sorted_indices = torch.sort(alphas_dict, descending=True)
num_indices = 10
top_indices_orig_dict = [dictionary[i] for i in sorted_indices[:num_indices]]
print("top coefficients: ", sorted_alphas[:num_indices].cpu().numpy())
alpha_ids = [pipe.tokenizer.decode(idx) for idx in top_indices_orig_dict]
print("top tokens: ", alpha_ids)

num_tokens = 50
alphas = torch.zeros(orig_embeddings.shape[0]).cuda()
sorted_alphas, sorted_indices = torch.sort(alphas_dict.abs(), descending=True)
top_word_idx = [dictionary[i] for i in sorted_indices[:num_tokens]]
for i, index in enumerate(top_word_idx):
    alphas[index] = alphas_dict[sorted_indices[i]]

# add placeholder for w^*
placeholder_token = '<>'
pipe.tokenizer.add_tokens(placeholder_token)
placeholder_token_id = pipe.tokenizer.convert_tokens_to_ids(placeholder_token)
pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
token_embeds = pipe.text_encoder.get_input_embeddings().weight.detach().requires_grad_(False)

# compute w^* and normalize its embedding
learned_embedding = torch.matmul(alphas, orig_embeddings).flatten()

# Calculate the average norm from original embeddings
norms = torch.norm(orig_embeddings, dim=1)
avg_norm = norms.mean().item()

# Normalize learned_embedding to have the average norm
learned_embedding /= learned_embedding.norm()
learned_embedding *= avg_norm

# add w^* to vocabulary
token_embeds[placeholder_token_id] = torch.nn.Parameter(learned_embedding)

# Define image grid function
def get_image_grid(images) -> Image:
    num_images = len(images)
    cols = int(math.ceil(math.sqrt(num_images)))
    rows = int(math.ceil(num_images / cols))
    width, height = images[0].size
    grid_image = Image.new('RGB', (cols * width, rows * height))
    for i, img in enumerate(images):
        x = i % cols
        y = i // cols
        grid_image.paste(img, (x * width, y * height))
    return grid_image

# Generate and display images for two prompts
prompt = 'a photo of a <>'

generator = torch.Generator("cuda").manual_seed(0)
image = pipe(prompt,
             guidance_scale=7.5,
             generator=generator,
             return_dict=False,
             num_images_per_prompt=6,
             num_inference_steps=50)
get_image_grid(image[0]).show()

prompt = 'a photo of a dog'
generator = torch.Generator("cuda").manual_seed(0)
image = pipe(prompt,
             guidance_scale=7.5,
             generator=generator,
             return_dict=False,
             num_images_per_prompt=6,
             num_inference_steps=50)
get_image_grid(image[0]).show()
