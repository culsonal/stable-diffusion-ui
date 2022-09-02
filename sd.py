
from diffusers import DDIMScheduler, LMSDiscreteScheduler, StableDiffusionPipeline
from tqdm.auto import tqdm
import numpy as np
import torch, cv2
import os


def get_models():
  pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4",
    revision="fp16", torch_dtype=torch.float16, use_auth_token=os.environ['SD_AUTH'])
  pipe.text_encoder.to('cpu')
  pipe.unet.to('cuda')
  pipe.vae.to('cuda')
  return { 'unet': pipe.unet, 'vae': pipe.vae, 'text_encoder': pipe.text_encoder, 'tokenizer': pipe.tokenizer }

@torch.autocast('cuda')
@torch.inference_mode()
def get_txt_embs(prompt, tokenizer, text_encoder, batch_size=1, device='cuda'):
  text_inps = tokenizer([ prompt ]*batch_size, padding='max_length',
    max_length=tokenizer.model_max_length, truncation=True, return_tensors='pt')
  uncond_input = tokenizer([''] * batch_size, padding='max_length',
    max_length=text_inps.input_ids.shape[-1], return_tensors='pt')
  text_embs   = text_encoder(text_inps.input_ids.to(device))[0]
  uncond_embs = text_encoder(uncond_input.input_ids.to(device))[0]
  text_embs = torch.cat([uncond_embs, text_embs])
  return text_embs.to(torch.float16).to('cuda')
  #return text_embs


@torch.autocast('cuda')
@torch.no_grad()
def get_im2im(inp_img, text_embs, unet, vae, num_inference_steps=50, guidance_scale=7.5, strength=.8):
  batch_size = inp_img.shape[0]

  scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
    beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
  scheduler.set_timesteps(num_inference_steps)

  offset, eta = 1, 0.
  scheduler.set_timesteps(num_inference_steps, offset=1)
 
  init_latents = vae.encode(inp_img).sample() * 0.18215
  init_latents = torch.cat([init_latents] * batch_size)

  init_timestep = min(num_inference_steps, int(num_inference_steps * strength) + offset)
  timesteps = scheduler.timesteps[-init_timestep]
  timesteps = torch.tensor([timesteps] * batch_size, dtype=torch.long, device='cuda')

  noise = torch.randn(init_latents.shape, device='cuda') * .9
  init_latents = scheduler.add_noise(init_latents, noise, timesteps)

  lats = init_latents
  t_start = max(num_inference_steps - init_timestep + offset, 0)
  for i, t in tqdm(enumerate(scheduler.timesteps[t_start:])):
    latent_model_input = torch.cat([lats] * 2)
    noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embs)["sample"]

    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    lats = scheduler.step(noise_pred, t, lats, eta=eta)["prev_sample"]

  out = vae.decode(lats * 1/0.18215)
  return (out*.5+.5).clamp(0, 1)[0].permute(0, 2, 1)


@torch.autocast('cuda')
@torch.no_grad()
def get_tex2im(text_embs, unet, vae, num_inference_steps=50, guidance_scale=7.5, dimw=448, dimh=448):
  batch_size = text_embs.shape[0]//2

  scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012,
    beta_schedule="scaled_linear", num_train_timesteps=1000)
  scheduler.set_timesteps(num_inference_steps)

  lats = torch.randn([ batch_size, 4, dimw//8, dimh//8 ], device='cuda')
  lats = lats * scheduler.sigmas[0]

  for i, t in tqdm(enumerate(scheduler.timesteps)):
    latent_model_input = torch.cat([lats] * 2)
    sigma = scheduler.sigmas[i]
    latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

    noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embs)["sample"]

    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    lats = scheduler.step(noise_pred, i, lats)["prev_sample"]

  out = vae.decode(lats * 1/0.18215)
  return (out*.5+.5).clamp(0, 1)[0].permute(0, 2, 1)
