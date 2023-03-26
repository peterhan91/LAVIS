import os
import pandas as pd
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from lavis.datasets.datasets.caption_datasets import CaptionDataset, CaptionEvalDataset


class MIMICCapDataset(CaptionDataset):
    def __init__(self, vis_processor, text_processor, vis_root, txt_path,
                 column='findings'):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        txt_path (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, 
                         txt_path, column)

        self.df = pd.read_csv(txt_path)
        self.col = column
        self.img_path = vis_root
        self.df = self.df.filter(items=['dicom_id', column]).dropna()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        dfs = self.df.iloc[index]
        img = Image.open(os.path.join(self.img_path, dfs['dicom_id']+'.jpg'))
        img = img.convert('RGB')
        txt = dfs[self.col]
        
        image = self.vis_processor(img)
        caption = self.text_processor(txt)

        return {
            "image": image,
            "text_input": caption,
            "image_id": dfs['dicom_id'],
        }        

class MIMICCapEvalDataset(CaptionEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, val_txt_path, 
                 column='findings'):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        val_txt_root (string): directory to store the validation annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, 
                         val_txt_path, column)

        self.df = pd.read_csv(val_txt_path)
        self.col = column
        self.img_path = vis_root
        self.df = self.df.filter(items=['dicom_id', column]).dropna()
            
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        dfs = self.df.iloc[index]
        img = Image.open(os.path.join(self.img_path, dfs['dicom_id']+'.jpg'))
        img = img.convert('RGB')
        
        image = self.vis_processor(img)

        return {
            "image": image,
            "image_id": dfs['dicom_id'],
            "instance_id": dfs['dicom_id'],
        }        

