import torch
import sys, os
from diffusers import StableDiffusionXLPipeline
import matplotlib.pyplot as plt
import random
# before running this script, make sure to set your terminal working directory as the project root directory.

TOTAL_SIZE, BATCH_SIZE = int(sys.argv[1]), int(sys.argv[2])
NUM_BATCHES = int(TOTAL_SIZE / BATCH_SIZE)
print(f"Total size: {TOTAL_SIZE}, number of batches: {NUM_BATCHES}, batch size: {BATCH_SIZE}", flush=True)
TYPE = float(sys.argv[3]); print(f"Type: {int(TYPE * 100)}", flush=True)
assert 0 <= TYPE <= 1, "Type must be between 0 and 1"
weights = [TYPE, 1 - TYPE]
print(torch.cuda.is_available(), flush=True)

PROFESSION = 'mathematics scientist'
positive_prompt = f"a photo of a {PROFESSION}, looking at the camera, ultra quality, sharp focus"
negative_prompt = "cartoon, anime, 3d, painting, b&w, low quality"
adapters = ["asian_female", "white_male"]

model_dict = {
    "asian_female": "NYUAD-ComNets/Asian_Female_Profession_Model",
    "white_male": "NYUAD-ComNets/White_Male_Profession_Model"
}
models = [model_dict[name] for name in adapters]
pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", variant="fp16", use_safetensors=True, torch_dtype=torch.float16).to("cuda")
for i,j in zip(models,adapters):
    pipeline.load_lora_weights(i, weight_name="pytorch_lora_weights.safetensors",adapter_name=j)
pipeline.set_adapters(adapters, adapter_weights=weights) # merge the two LoRA checkpoints
generator = torch.Generator(device="cuda")
generator.manual_seed(0)
torch.cuda.empty_cache()

os.makedirs(f"out/merged/{int(TYPE * 100)}", exist_ok=True) # create the output directory if it doesn't exist
# save the images in batches, with the correct order
for i in range(NUM_BATCHES):
    if i % 50 == 0:
        print(f"Generating batch {i}...", flush=True)
    images = pipeline(
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps = 100,
        num_images_per_prompt = BATCH_SIZE,
        generator = generator,
        height = 1024,
        width = 1024
    ).images
    torch.cuda.empty_cache()
    for j, img in enumerate(images):
        img.save(f"out/merged/{int(TYPE * 100)}/{i * BATCH_SIZE + j}.png")