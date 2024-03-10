from diffusers import DDPMPipeline, DDIMPipeline

# load model and scheduler
pipe = DDIMPipeline.from_pretrained("google/ddpm-cat-256")
unet = pipe.unet
import torch

# ran_img = torch.randn(1, 3, 256, 256)
import PIL
img = PIL.Image.open("/data/tmp_teja/DiffusionFlareRemoval/cat.jpg")

from torchvision import transforms

preprocess = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


ran_img = preprocess(img).unsqueeze(0)
# print(unet(ran_img, torch.tensor([10])).sample.shape)



