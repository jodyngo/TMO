import os
from PIL import Image
import torchvision as tv
from .transforms import *


class TestYTOBJ(torch.utils.data.Dataset):
    def __init__(self, root):
        classes = sorted(os.listdir(root))
        self.seqs = []
        for cls in classes:
            seqs = sorted(os.listdir(root + '/' + cls))
            for seq in seqs:
                self.seqs.append(root + '/' + cls + '/' + seq)
        self.to_tensor = tv.transforms.ToTensor()

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        img_list = sorted(os.listdir(self.seqs[idx] + '/JPEGImages'))
        flow_list = sorted(os.listdir(self.seqs[idx] + '/JPEGFlows'))
        mask_list = sorted(os.listdir(self.seqs[idx] + '/Annotations'))

        # find valid frames
        valid_frames = []
        for mask in mask_list:
            frame_num = int(mask.split('.png')[0])
            valid_frames.append(frame_num - 1)

        # generate testing snippets
        imgs = []
        flows = []
        masks = []
        for i in range(len(img_list)):
            img = Image.open(self.seqs[idx] + '/JPEGImages' + '/' + img_list[i])
            img = img.resize((384, 384), Image.BICUBIC)
            imgs.append(self.to_tensor(img))
        for i in range(len(flow_list)):
            flow = Image.open(self.seqs[idx] + '/JPEGFlows' + '/' + flow_list[i])
            flow = flow.resize((384, 384), Image.BICUBIC)
            flows.append(self.to_tensor(flow))
        for i in range(len(mask_list)):
            mask = Image.open(self.seqs[idx] + '/Annotations' + '/' + mask_list[i]).convert('L')
            mask = mask.resize((384, 384), Image.NEAREST)
            masks.append(self.to_tensor(mask))

        # gather all frames
        imgs = torch.stack(imgs, dim=0)
        flows = torch.stack(flows, dim=0)
        masks = torch.stack(masks, dim=0)
        masks = (masks > 0.5).long()
        return {'imgs': imgs, 'flows': flows, 'masks': masks, 'path': self.seqs[idx], 'valid_frames': valid_frames}
