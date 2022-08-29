# Stable diffusion UI

Rapid feedback stable diffusion UI.

![coolll](https://user-images.githubusercontent.com/112416131/187251967-3e65e38e-b5fb-4f46-a194-0e2b0900a381.gif)

Reads the prompt from `prompt.txt`. Click the image to draw on it. Press `escape` to apply text2image from your prompt. Press `space` to apply im2im on the current image. Press `enter` to activate your camera and load a frame into the image: fun in combination with im2im. (Only the first line of the prompt file is used to make it convenient to swap in and out useful prompts.)


#### Install
Run
```
!pip install omegaconf einops pytorch-lightning transformers kornia
!pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
!pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip
!pip install -U git+https://github.com/huggingface/diffusers.git
!pip install taichi
```

Then set the environment variable `SD_AUTH` to your hugginface token that you get from here https://huggingface.co/settings/tokens.

#### Run
Just `git clone https://github.com/culsonal/stable-diffusion-ui`, then cd into the repo.

Then run `python stable-ui.py`.

#### Additional notes
* Currently customized for 4 gb vram which means the text encoder is placed on the CPU (which doesn't affect performance much) and the default width and height is 448x448.
* Early stage prototype, lots of fun stuff to improve.
