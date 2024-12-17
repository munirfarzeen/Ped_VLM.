from torch.utils.data import Dataset
from torchvision.io import read_image
import os
import json
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PedDataset(Dataset):

    def __init__(self, input_file, config, tokenizer, transform=None):
        with open(input_file) as f:
            self.data = json.load(f)

        self.config = config
        self.tokenizer = tokenizer
        self.transform = transform
        if self.config.encoder=='clip':
            self.precossor=CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the question and answer at the idx
        qa, img_path = self.data[idx]
        img_path = list(img_path.values())

        q_text, a_text ,i_label= qa['Q'], qa['A'],qa['C']
        q_text = f"Question: {q_text} Answer:"

        #Concatenate images into a single tensor
        if self.config.encoder=='clip':

            # resize_transform = transforms.Resize((224, 224))
            rgb_image=Image.open(os.path.join(self.config.img_path,img_path[0]))
            rgb_image=self.precossor(images=rgb_image,return_tensors="pt")
            rgb_image={k: v.squeeze(0) for k, v in rgb_image.items()}
            e_im=rgb_image['pixel_values'].to(device) 
            # rgb_image=resize_transform(rgb_image)
            if self.config.optical:
                opticalflow=Image.open(os.path.join(self.config.img_path,img_path[1])).convert('RGB')
                opticalflow=self.precossor(images=opticalflow,return_tensors="pt")
                opticalflow={k: v.squeeze(0) for k, v in opticalflow.items()}
                op=opticalflow['pixel_values'].to(device) 
                imgs=[e_im,op]
            else:
                imgs=[e_im]
        else:
        
            rgb_image=self.transform(read_image(os.path.join(self.config.img_path,img_path[0])).float()).to(device)
            if self.config.optical:
                opticalflow=read_image(os.path.join(self.config.img_path,img_path[1]))
                new_channel = (opticalflow[2, :, :] + opticalflow[3, :, :]) / 2.0
                opticalflow=self.transform(torch.stack((opticalflow[0, :, :], opticalflow[1, :, :], new_channel))).float().to(device) 
                imgs=[rgb_image,opticalflow]
            else:
                imgs=[rgb_image]
        
        imgs = torch.stack(imgs, dim=0)


        # imgs = [self.transform(os.path.join(self.config.data_path, img_path(p)).float()).to(device) for p in img_path]
        # imgs=[self.transform(read_image((os.path.join(self.config.data_path,p))).float()).to(device) for p in img_path]
        # [6,3,224,224]
            
        i_label=torch.tensor(i_label).to(device)

        return q_text, imgs, a_text, i_label,sorted(list(img_path))
    
    # def __getitem__(self, idx):
    #     # Get the question and answer at the idx
    #     qa, img_path = self.data[idx]
    #     img_path = list(img_path.values())

    #     q_text, a_text ,i_label= qa['Q'], qa['A'],qa['C']
    #     q_text = f"Question: {q_text} Answer:"

    #     #Concatenate images into a single tensor
    #     if self.config.encoder=='clip':
    #         opticalflow=Image.open(os.path.join(self.config.img_path,img_path[1])).convert('RGB')
    #         opticalflow=self.precossor(images=opticalflow,return_tensors="pt")
    #         opticalflow={k: v.squeeze(0) for k, v in opticalflow.items()}
    #         op=opticalflow['pixel_values'].to(device) 
            

    #         # resize_transform = transforms.Resize((224, 224))
    #         # rgb_image=Image.open(os.path.join(self.config.img_path,img_path[0]))
    #         # rgb_image=self.precossor(images=rgb_image,return_tensors="pt")
    #         # rgb_image={k: v.squeeze(0) for k, v in rgb_image.items()}
    #         # e_im=rgb_image['pixel_values'].to(device) 
    #         # rgb_image=resize_transform(rgb_image)
    #         if self.config.optical:
    #             # opticalflow=Image.open(os.path.join(self.config.img_path,img_path[1])).convert('RGB')
    #             # opticalflow=self.precossor(images=opticalflow,return_tensors="pt")
    #             # opticalflow={k: v.squeeze(0) for k, v in opticalflow.items()}
    #             # op=opticalflow['pixel_values'].to(device) 

    #             rgb_image=Image.open(os.path.join(self.config.img_path,img_path[0]))
    #             rgb_image=self.precossor(images=rgb_image,return_tensors="pt")
    #             rgb_image={k: v.squeeze(0) for k, v in rgb_image.items()}
    #             e_im=rgb_image['pixel_values'].to(device) 
    #             imgs=[e_im,op]
    #         else:
    #             # imgs=[e_im]
    #             imgs=[op]
    #     else:
        
    #         rgb_image=self.transform(read_image(os.path.join(self.config.img_path,img_path[0])).float()).to(device)
    #         if self.config.optical:
    #             opticalflow=read_image(os.path.join(self.config.img_path,img_path[1]))
    #             new_channel = (opticalflow[2, :, :] + opticalflow[3, :, :]) / 2.0
    #             opticalflow=self.transform(torch.stack((opticalflow[0, :, :], opticalflow[1, :, :], new_channel))).float().to(device) 
    #             imgs=[rgb_image,opticalflow]
    #         else:
    #             imgs=[rgb_image]
        
    #     imgs = torch.stack(imgs, dim=0)


    #     # imgs = [self.transform(os.path.join(self.config.data_path, img_path(p)).float()).to(device) for p in img_path]
    #     # imgs=[self.transform(read_image((os.path.join(self.config.data_path,p))).float()).to(device) for p in img_path]
    #     # [6,3,224,224]
            
    #     i_label=torch.tensor(i_label).to(device)

    #     return q_text, imgs, a_text, i_label,sorted(list(img_path))


    def collate_fn(self, batch):
        q_texts, imgs, a_texts, i_label, _ = zip(*batch)
        imgs = torch.stack(list(imgs), dim=0)
        i_label = torch.stack(list(i_label), dim=0)

        encodings = self.tokenizer(q_texts, padding=True, return_tensors="pt").input_ids.to(device)
        labels = self.tokenizer(a_texts, padding=True, return_tensors='pt').input_ids.to(device)
        

        return encodings, imgs, labels,i_label

    def test_collate_fn(self, batch):
        q_texts, imgs, a_texts,i_labels, img_path = zip(*batch)
        imgs = torch.stack(list(imgs), dim=0)
        i_labels = torch.stack(list(i_labels), dim=0)

        encodings = self.tokenizer(q_texts, padding=True, return_tensors="pt").input_ids.to(device)
        labels = self.tokenizer(a_texts, padding=True, return_tensors='pt').input_ids.to(device)
       

        return list(q_texts), encodings, imgs, labels,i_labels, img_path
