from copy import deepcopy
import torch
from diffusers import StableDiffusionXLPipeline
import matplotlib.pyplot as plt
import random
import sys, os
# import custom inference & postprocessing functions we wrote.
# before running this script, make sure to set your terminal working directory as the project root directory.
from src.SDXL_inference import ScoreFusion_inference, postprocess_image

TOTAL_SIZE, BATCH_SIZE = int(sys.argv[1]), int(sys.argv[2])
W1_VALUE = float(sys.argv[3])
print(f"W1 value: {W1_VALUE}")

# if W1_VALUE == 0.5:
#     subpath = 'kl' # original kl experiment
# else:
subpath = f'lambdas/{int(W1_VALUE * 100)}' # ablation studies with different lambda values

NUM_BATCHES = int(TOTAL_SIZE / BATCH_SIZE)
print(f"Total size: {TOTAL_SIZE}, number of batches: {NUM_BATCHES}, batch size: {BATCH_SIZE}")

PROFESSION = 'mathematics scientist'
positive_prompt = f"a photo of a {PROFESSION}, looking at the camera, ultra quality, sharp focus"
negative_prompt = 'cartoon, anime, 3d, painting, b&w, low quality'
w1 = torch.tensor(W1_VALUE, device="cuda", dtype=torch.float16)
w2 = torch.tensor(1 - W1_VALUE, device="cuda", dtype=torch.float16)
adapters=["asian_female", "white_male"]

model_dict = {
    "asian_female": "NYUAD-ComNets/Asian_Female_Profession_Model",
    "white_male": "NYUAD-ComNets/White_Male_Profession_Model"
}
models = [model_dict[name] for name in adapters]
pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", variant="fp16", use_safetensors=True, torch_dtype=torch.float16).to("cuda")
for i,j in zip(models,adapters):
    pipeline.load_lora_weights(i, weight_name="pytorch_lora_weights.safetensors",adapter_name=j)

pipeline.set_adapters(adapters, adapter_weights=[1.0, 0.0])
unet1 = deepcopy(pipeline.unet).cuda()
pipeline.set_adapters(adapters, adapter_weights=[0.0, 1.0])
unet2 = deepcopy(pipeline.unet).cuda()
pipeline.set_adapters(adapters, adapter_weights=[0.0, 0.0]) # unset LoRA adapters

# Note: this is just for 'warmstarting' the pipeline, so disregard the images.
_ = pipeline(prompt=positive_prompt,negative_prompt=negative_prompt,
        num_inference_steps=2, num_images_per_prompt=1).images

generator = torch.Generator(device="cuda")
generator.manual_seed(0) # set seed for reproducibility
torch.cuda.empty_cache()

os.makedirs(f"out/{subpath}", exist_ok=True) # create the output directory if it doesn't exist
all_latents = []
for i in range(NUM_BATCHES):
    images_latents = ScoreFusion_inference(pipeline=pipeline,
        l1 = w1,
        l2 = w2,
        unet1 = unet1,
        unet2 = unet2,
        num_images_per_prompt = BATCH_SIZE,
        num_inference_steps = 100,
        prompt = positive_prompt,
        negative_prompt = negative_prompt,
        height = 1024,
        width = 1024,
        generator = generator,
        output_type = 'latent' # don't use VAE in this step; do it outisde the loop explicitly
    ).images
    all_latents.append(images_latents)
    torch.cuda.empty_cache()
    images = [postprocess_image(pipeline, latent.unsqueeze(0)).images[0] for latent in images_latents]
    for j, img in enumerate(images):
        img.save(f"out/{subpath}/{i * BATCH_SIZE + j}.png")
