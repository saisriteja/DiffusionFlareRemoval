import os, io, csv, math, random
import numpy as np
from einops import rearrange
from decord import VideoReader

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
# import albumentations as A
from flare_distortion import flare_distortion
import cv2

class WebVid10M(Dataset):
    def __init__(
            self,
            csv_path, video_folder,
            sample_size=256, sample_stride=4, sample_n_frames=16,
            is_image=False,
        ):
        # zero_rank_print(f"loading annotations from {csv_path} ...")
        with open(csv_path, 'r') as csvfile:
            self.dataset = list(csv.DictReader(csvfile))
        self.length = len(self.dataset)
        # zero_rank_print(f"data scale: {self.length}")

        self.video_folder    = video_folder
        self.sample_stride   = sample_stride
        self.sample_n_frames = sample_n_frames
        self.is_image        = is_image
        
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        
        self.pixel_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(sample_size[0]),
            transforms.CenterCrop(sample_size),
            # rain and snow
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

    
    def get_batch(self, idx):
        video_dict = self.dataset[idx]
        videoid, name, page_dir = video_dict['videoid'], video_dict['name'], video_dict['page_dir']
        
        video_dir    = os.path.join(self.video_folder, f"{videoid}.mp4")
        video_reader = VideoReader(video_dir)
        video_length = len(video_reader)
        print("video length", video_length)

        
        if not self.is_image:
            clip_length = min(video_length, (self.sample_n_frames - 1) * self.sample_stride + 1)
            start_idx   = random.randint(0, video_length - clip_length)
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
        else:
            batch_index = [random.randint(0, video_length - 1)]

        # convert to PIL
        data = video_reader.get_batch(batch_index).asnumpy()
        # data = np.array([self.dist_transform(image=img)['image'] for img in data])

        print("started flare distortion")

        flare_vid, base_vid, merge_vid = flare_distortion(data)

        # print("added flare distortion")

        # f = f.permute(1,2,0)
        # f = f.numpy()
        

        # data = self.dist_transform(image=data[0])['image']

        # for i in range(len(data)):
        #     cv2.imwrite(f"distorted_{i}.png", data[i])


        # pixel_values = torch.from_numpy(data).permute(0, 3, 1, 2).contiguous()
        # pixel_values = pixel_values / 255.

        # convert a list of torch tensors to a single tensor
        flare_vid = torch.stack(flare_vid)
        base_vid = torch.stack(base_vid)
        merge_vid = torch.stack(merge_vid)


        # for i in range(len(flare_vid)):
        #     img_tmp = merge_vid[i].permute(1,2,0)
        #     img_tmp = img_tmp.numpy()
        #     print("img tmp",img_tmp.shape)
        #     img_tmp = img_tmp * 255
        #     cv2.imwrite(f"distorted_flare_{i}.png", img_tmp)

        print("stacked the tensors")

        pixel_values = merge_vid
        del video_reader

        if self.is_image:
            pixel_values = pixel_values[0]
        
        return pixel_values, name

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, name = self.get_batch(idx)
                break

            except Exception as e:
                idx = random.randint(0, self.length-1)
        pixel_values = self.pixel_transforms(pixel_values)
        sample = dict(pixel_values=pixel_values, text=name)
        # print(sample.shape)
        # print(sample["pixel_values"].shape)

        print(sample["pixel_values"].shape, len(sample["text"]))

        for no,img in enumerate(sample["pixel_values"]):
            print(img.shape)
            img = img.permute(1,2,0)
            img = (img + 1.0) / 2.0

            # get the values between 0 to 1
            img = img.numpy()

            # print min and max 
            img = img * 255
            print(np.min(img), np.max(img))
            cv2.imwrite(f"distorted_Ings_{no}.png", img)


        return sample


import torchvision
import imageio


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)

def distortion(data):
    return data

if __name__ == "__main__":
    # from animatediff.utils.util import save_videos_grid

    print("starting the dataset")

    dataset = WebVid10M(
        csv_path="truncated_results_2M_val.csv",
        video_folder="vids/",
        sample_size=512,
        sample_stride=4, 
        sample_n_frames=6,
        is_image=False,
    )
    # import pdb
    # pdb.set_trace()
    

    print("making the dataloader")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=4,)


    data = next(iter(dataloader))
    print(data["pixel_values"].shape, len(data["text"]))



    for idx, batch in enumerate(dataloader):
        print(batch["pixel_values"].shape, len(batch["text"]))
        print(batch["pixel_values"].shape)
        # for no,img in enumerate(batch["pixel_values"][0]):
        #     img = img.permute(1,2,0)
        #     img = (img + 1.0) / 2.0

        #     # get the values between 0 to 1
        #     img = img.numpy()

        #     # print min and max 
        #     img = img * 255
        #     print(np.min(img), np.max(img))
        #     cv2.imwrite(f"distorted_Ings_{no}.png", img)


        # for i in range(batch["pixel_values"].shape[0]):
        #     save_videos_grid(batch["pixel_values"][i:i+1].permute(0,2,1,3,4), os.path.join(".", f"{idx}-{i}.mp4"), rescale=True)

        break
