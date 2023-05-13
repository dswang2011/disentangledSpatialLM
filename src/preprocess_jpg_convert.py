
import datasets
from datasets import Dataset, Features, Sequence, Value, Array2D, Array3D
import os
import random
from OCRs import tesseract4img
from utils import util
from datasets import concatenate_datasets,load_from_disk


def img_fix(sample):
    sample['image'] = sample['image'].replace('cdip_v1','cdip_vx').replace('.tif','.jpg')
    return sample

def img_exist(sample):
    return os.path.exists(sample['image'])


def convert_and_save(sample):
    img_path = sample['image']
    img_obj,_ = tesseract4img._load_image(img_path)
    if not img_obj: return False

    # create new dir
    strs = img_path.split('/')
    dir = '/'.join(strs[:-1])
    dir = dir.replace('cdip_v1','cdip_vx')
    filename = strs[-1]
    # summarize tgt path

    if not os.path.exists(dir):
        # Create a new directory because it does not exist
        os.makedirs(dir)
    try:
        save_path = os.path.join(dir, filename.replace('.tif','.jpg'))
        image = img_obj.convert("RGB")
        image.save(save_path, "JPEG", quality=80)
        return True
    except Exception as e:
        print('skipped:',e)
        return False


# if __name__ == '__main__':
#     # load dataset
#     for i in range(6,8):
#         ds_path = '/home/ubuntu/air/vrdu/datasets/rvl_HF_datasets/full_cdip_a'+str(i)+'_dataset.hf'
#         raw_ds = load_from_disk(ds_path)
#         print(raw_ds)
#         # filter dataset; 
#         tgt_ds = raw_ds.filter(convert_and_save)
#         print(tgt_ds)
#         # save into another one
#         saveto = ds_path + 'filtered'
#         tgt_ds.save_to_disk(saveto)

# merge dataset
if __name__ == '__main__':
    comb_ds = []
    for i in range(7):
        ds_path = '/home/ubuntu/air/vrdu/datasets/rvl_HF_datasets/full_cdip_a'+str(i)+'_dataset.hffiltered'
        raw_ds = load_from_disk(ds_path)
        comb_ds.append(raw_ds)
        print(raw_ds)

    saveto = 'a_comb.hf'
    comb_ds = concatenate_datasets(comb_ds)
    print('combined:', comb_ds)
    comb_ds = comb_ds.map(img_fix)
    comb_ds = comb_ds.filter(img_exist)
    print('verified:',comb_ds)
    comb_ds.save_to_disk(saveto)
