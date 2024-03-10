import numpy as np
import torchvision.transforms as transforms


import torchvision.transforms.functional as TF
import random


class RandomGammaCorrection(object):
	def __init__(self, gamma = None):
		self.gamma = gamma
	def __call__(self,image):
		if self.gamma == None:
			# more chances of selecting 0 (original image)
			gammas = [0.5,1,2]
			self.gamma = random.choice(gammas)
			return TF.adjust_gamma(image, self.gamma, gain=1)
		elif isinstance(self.gamma,tuple):
			gamma=random.uniform(*self.gamma)
			return TF.adjust_gamma(image, gamma, gain=1)
		elif self.gamma == 0:
			return image
		else:
			return TF.adjust_gamma(image,self.gamma,gain=1)


def remove_background(image):
	#the input of the image is PIL.Image form with [H,W,C]
	image=np.float32(np.array(image))
	_EPS=1e-7
	rgb_max=np.max(image,(0,1))
	rgb_min=np.min(image,(0,1))
	image=(image-rgb_min)*rgb_max/(rgb_max-rgb_min+_EPS)
	image=torch.from_numpy(image)
	return image


gamma=np.random.uniform(1.8,2.2)
to_tensor=transforms.ToTensor()
adjust_gamma=RandomGammaCorrection(gamma)
adjust_gamma_reverse=RandomGammaCorrection(1/gamma)
# color_jitter=transforms.ColorJitter(brightness=(0.8,3),hue=0.0)


from PIL import Image
import torch
from torch.distributions import Normal


from glob import glob
import cv2

def flare_distortion(base_imgs,
					 flare_path = '/media/cilab/data/NTIRE/flare/Flare7Kpp/Flare7K/Scattering_Flare/Compound_Flare',
					 reflective_path = '/media/cilab/data/NTIRE/flare/Flare7Kpp/Flare7K/Reflective_Flare',
					 flare_no= 0,
					 reflective_no = 0):
    transform_flare=transforms.Compose([transforms.RandomAffine(degrees=(0,360),
																scale=(0.8,1.5),
																translate=(300/1440,300/1440),
																shear=(-20,20)),
                                                                transforms.CenterCrop((512,512)),
                                                                transforms.RandomHorizontalFlip(),
                                                                transforms.RandomVerticalFlip()
                                ])

    flare_list = glob(flare_path + '/*')
    flare_path=flare_list[flare_no]
    flare_img =Image.open(flare_path)
    
    reflective_list = glob(reflective_path + '/*')
    reflective_path=reflective_list[reflective_no]
    reflective_img =Image.open(reflective_path)
    flare_img=to_tensor(flare_img)
    flare_img=adjust_gamma(flare_img)
    reflective_img=to_tensor(reflective_img)
    reflective_img=adjust_gamma(reflective_img)
    flare_img = torch.clamp(flare_img+reflective_img,min=0,max=1)
    flare_img=remove_background(flare_img)

    flare_vid = []
    base_vid = []
    merge_vid = []


    flare_img=transform_flare(flare_img)

    #flare blur
    blur_transform=transforms.GaussianBlur(21,sigma=(0.1,3.0))
    flare_img=blur_transform(flare_img)


    for no,base_img in enumerate(base_imgs):
        # resize it to 512,512
        # base_img=base_img.resize((512,512))
		# resize using cv2
        base_img = cv2.resize(base_img, (512, 512))
        base_img=to_tensor(base_img)
        base_img=adjust_gamma(base_img)
        sigma_chi=0.01*np.random.chisquare(df=1)
        base_img=Normal(base_img,sigma_chi).sample()
        gain=np.random.uniform(0.5,1.2)
        flare_DC_offset=np.random.uniform(-0.02,0.02)
        base_img=gain*base_img
        base_img=torch.clamp(base_img,min=0,max=1)

        # keep the flare img rotated
        flare_img = torch.roll(flare_img, shifts=30, dims=2)
        flare_img=flare_img+flare_DC_offset
        flare_img=torch.clamp(flare_img,min=0,max=1)
        #merge image
        merge_img=flare_img+base_img
        merge_img=torch.clamp(merge_img,min=0,max=1)

        flare_vid.append(adjust_gamma_reverse(flare_img))
        base_vid.append(adjust_gamma_reverse(base_img))
        merge_vid.append(adjust_gamma_reverse(merge_img))

    # return adjust_gamma_reverse(base_img),adjust_gamma_reverse(flare_img),adjust_gamma_reverse(merge_img)
    return flare_vid, base_vid, merge_vid











# from glob import glob
# import cv2
# path = '/home/saiteja/flare_IITM_Research/datasets/Flickr24K'
# imgs = glob(path + '/*')
# img = imgs[0]
# base_img= cv2.imread(img)
# base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)
# base_imgs = [base_img] * 5
# flare, base, merge = flare_distortion(base_imgs)
# # print(len(flare), flare[0].shape)

# import cv2
# for no, f in enumerate(merge):
#     f = f.permute(1,2,0)
#     f = f.numpy()
#     # save the image
#     f = f * 255
#     cv2.imwrite(f'flare_{no}.png', f)





