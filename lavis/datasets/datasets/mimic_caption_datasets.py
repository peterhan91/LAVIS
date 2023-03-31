import os
import random
import pandas as pd
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from lavis.datasets.datasets.caption_datasets import CaptionDataset, CaptionEvalDataset


class MIMICCapDataset(CaptionDataset):
    def __init__(self, vis_processor, text_processor, vis_root, txt_path, column):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        txt_path (string): directory to store the annotation file
        """
        # super().__init__(vis_processor, text_processor, vis_root, txt_path, column)

        self.df = pd.read_csv(txt_path)
        self.col = column
        self.img_path = vis_root
        self.df = self.df.filter(items=['dicom_id', column]).dropna()
        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        dfs = self.df.iloc[index]
        img = Image.open(os.path.join(self.img_path, dfs['dicom_id']+'.jpg'))
        img = img.convert('RGB')
        txt = dfs[self.col]
        # txt_in = dfs['indication']
        words = txt.split()
        mid = random.randint(int(0.1*len(words)), int(0.7*len(words)))
        txt_in = ' '.join(words[:mid])
        txt_out = ' '.join(words[mid:]) 
        
        image = self.vis_processor(img)
        caption_output = self.text_processor(txt_out)
        caption_input = self.text_processor(txt_in)

        return {
            "image": image,
            "text_input": caption_input,         # upper part of findings
            "text_output": caption_output,       # lower part of findings
            "image_id": dfs['dicom_id'],
        }        

class MIMICCapEvalDataset(CaptionEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, txt_path, column):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        val_txt_root (string): directory to store the validation annotation file
        """
        # super().__init__(vis_processor, text_processor, vis_root, txt_path, column)

        self.df = pd.read_csv(txt_path)
        self.col = column
        self.img_path = vis_root
        self.df = self.df.filter(items=['dicom_id', 'indication', column]).dropna()
        self.vis_processor = vis_processor
        self.text_processor = text_processor
            
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        dfs = self.df.iloc[index]
        img = Image.open(os.path.join(self.img_path, dfs['dicom_id']+'.jpg'))
        img = img.convert('RGB')
        
        image = self.vis_processor(img)

        return {
            "image": image,
            # "text_input": dfs['indication'],
            "text_input": 'What are the clinical findings of this patient? Answer:',
            "image_id": dfs['dicom_id'],
            "instance_id": dfs['dicom_id'],
        }        

