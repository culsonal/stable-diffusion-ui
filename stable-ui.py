from sd import get_models, get_im2im, get_tex2im, get_txt_embs
from taichi.math import vec2, vec3, length, normalize
import taichi as ti; ti.init(arch=ti.cpu)
import numpy as np
import torch, cv2

def run_im2im(conf, num_inference_steps=50, guidance_scale=13.5, strength=.25):
  txt_emb = get_txt_embs(get_prompt_from_file(), conf['tokenizer'], conf['text_encoder'], 1, 'cpu')
  inp_img = (torch.flip(draw.to_torch('cuda').permute(0, 2, 1), [1]).unsqueeze(0) - .5)*2
  out = get_im2im(inp_img, txt_emb, conf['unet'], conf['vae'], num_inference_steps, guidance_scale, strength)
  draw.from_torch(torch.flip(out, [2]))

def run_tex2im(conf, num_inference_steps=50, guidance_scale=7.5):
  txt_emb = get_txt_embs(get_prompt_from_file(), conf['tokenizer'], conf['text_encoder'], 1, 'cpu')
  out = get_tex2im(txt_emb, conf['unet'], conf['vae'], num_inference_steps, guidance_scale, dimw, dimh)
  draw.from_torch(torch.flip(out, [2]))

def get_prompt_from_file():
  with open('prompt.txt') as f:
    return f.readlines()[0]

def get_sd_setup():
  ret = get_models()
  unet, vae, text_encoder, tokenizer = ret['unet'], ret['vae'], ret['text_encoder'], ret['tokenizer']
  ret['text_encoder'].to('cpu').to(torch.float32)
  return { 'text_encoder': text_encoder, 'tokenizer': tokenizer, 'unet': unet, 'vae': vae }

def grab_cam(cam):
  if cam == None: cam = cv2.VideoCapture(0)
  ret, frame = cam.read()
  frame = np.flip(frame.astype(np.float16)/255., 0)[...,::-1]
  draw.from_numpy(np.transpose(frame[:dimw, :dimh, :], (2, 1, 0)))
  return cam

@ti.kernel
def handle_input(mx:ti.f16, my:ti.f16):
  for c, x, y in draw:
    if length(vec2(x/dimw-mx, y/dimh-my)) < .01:
      draw[c, x, y] = 0

@ti.kernel
def init():
  for x, y, c in ti.ndrange(dimw, dimh, 3):
    draw[c, x, y] = 1

@ti.kernel
def render():
  for x, y, c in ti.ndrange(ddimw, ddimh, 3):
    xx, yy = int(x/ddimw*dimw), int(y/ddimh*dimh)
    display[x, y, c] = draw[c, xx, yy]

cam = None
dimw, dimh = 64*7, 64*7
ddimw, ddimh = int(dimw*1.7), int(dimh*1.7)
draw      = ti.field(ti.f16, (3, dimw, dimh))
display   = ti.field(ti.f16, (ddimw, ddimh, 3))
sd_models = get_sd_setup()

init(); gui = ti.GUI('', (ddimw, ddimh))
cfg_slider       = gui.slider('guidance scale', 5, 20, step=.1); cfg_slider.value = 7.5
iters_slider     = gui.slider('num iters', 1, 100, step=1);      iters_slider.value = 50
strength_slider  = gui.slider('im2im strength', 0, 1, step=.01); strength_slider.value = .25
while True:
  render(); gui.set_image(display); gui.show()
  for e in gui.get_events(gui.PRESS):
    if e.key == ti.GUI.SPACE:  run_im2im(sd_models,  int(iters_slider.value), cfg_slider.value, strength_slider.value)
    if e.key == ti.GUI.ESCAPE: run_tex2im(sd_models, int(iters_slider.value), cfg_slider.value)
    if e.key == ti.GUI.RETURN: cam = grab_cam(cam)
  if gui.is_pressed(ti.GUI.LMB): handle_input(*gui.get_cursor_pos())

