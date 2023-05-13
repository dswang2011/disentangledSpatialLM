import datasets
from datasets import Dataset, Features, Sequence, Value, Array2D, Array3D
import os
import random
from OCRs import tesseract4img
from utils import util
from datasets import concatenate_datasets,load_from_disk

'''
what I wanna do:
    1 Load all images 
    2 Generate dataset dict as dataset file
    3 load them for pretraining 
'''


# step1: get all images from a folder
images = '/home/ubuntu/air/vrdu/datasets/images'


'''

SROIE dataset (receipt);


0 letter; 15
1 form; 10
2 email; 2
3 handwritten; 1
4 advertisement; 2.5
5, scientific report; 4.5
6 scientific publication; 2
7 specification; 5
8 file folder; 1
9 news article; 1.5
10 budget; 7
11 invoice, 9.5
12 presentation (slide): 2
13 questionaire; 7
14 resume; 3
15 memo; 1.5


'''
tgt_prob = {
    '0':0.15,
    '1':1.0,
    '2':0.15,
    '3':0.15,
    '4':0.25,
    '5':0.45,
    '6':0.2,
    '7':0.55,
    '8':0.1,
    '9':0.15,
    '10':0.7,
    '11':0.95,
    '12':0.15,
    '13':0.7,
    '14':0.3,
    '15':0.15,
}

id2label = {
    '0':'a0_letter',
    '1':'b1_form',
    '2':'c2_email',
    '3':'d3_handwritten',
    '4':'e4_advertisement',
    '5':'f5_sci_paper',
    '6':'g6_sci_report',
    '7':'h7_specification',
    '8':'i8_file_folder',
    '9':'j9_news',
    '10':'k10_budget',
    '11':'l11_invoice',
    '12':'m12_presentation',
    '13':'n13_questionaire',
    '14':'o14_resume',
    '15':'p15_memo',
}

label2id = {label:id for id,label in id2label.items()}


# sampling some data proportionally to the the ratio above;
def get_ratioly_sampled(imgs,labels):
    sub_imgs, sub_labels = [],[]
    for img,label in zip(imgs,labels):
        prob = random.random()
        label = label2id[label] # get id
        if prob > tgt_prob[label]: continue
        sub_imgs.append(img)
        sub_labels.append(label)
    return sub_imgs, sub_labels


# Step1. get image path list
def get_img_label_pairs(split='train'):  # .tif files
    imgs = []
    labels = []
    if split=='train':
        split_file = '/home/ubuntu/air/vrdu/datasets/labels/train.txt'
    elif split=='val':
        split_file = '/home/ubuntu/air/vrdu/datasets/labels/val.txt'
    elif split=='test':
        split_file = '/home/ubuntu/air/vrdu/datasets/labels/test.txt'
    else:
        print('wrong split:',split)
        return

    with open(split_file, 'r', encoding='utf8') as fr:
        data = fr.readlines()
    for row in data:
        path, label = row.split(' ')
        label = label.strip()
        labels.append(label)
        image_path = os.path.join(images, path)
        imgs.append(image_path)
    return imgs,labels


# step1. get image path list from a folder
funsd_plus_val = '/home/ubuntu/air/vrdu/datasets/funsd_plus/val_data/images'
funsd_plus_train = '/home/ubuntu/air/vrdu/datasets/funsd_plus/train_data/images'
funsd_plus_test = '/home/ubuntu/air/vrdu/datasets/funsd_plus/test_data/images'

cord_train = '/home/ubuntu/air/vrdu/datasets/CORD/train/image'
cord_test = '/home/ubuntu/air/vrdu/datasets/CORD/test/image'
cord_dev = '/home/ubuntu/air/vrdu/datasets/CORD/dev/image'

sorie_train = '/home/ubuntu/air/vrdu/datasets/sorie2019/train/img'
sorie_test = '/home/ubuntu/air/vrdu/datasets/sorie2019/test/img'


def get_imgs_list(dir):
    files = os.listdir(dir)
    files = [os.path.join(dir,file) for file in files]
    return files

def get_imgs_dfs(dir, suffix = 'tif'):
    res = []
    files = os.listdir(dir)
    for file in files:
        file_path = os.path.join(dir,file)

        if os.path.isdir(file_path):
            sub_res = get_imgs_dfs(file_path, suffix)
            res += sub_res
        elif file.endswith(suffix):
            res.append(file_path)
    return res


# step2: shuffle and split into 5 subsets
def _split(input_list, n):
    k, m = divmod(len(input_list), n)
    return (input_list[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


# load file list
# e.g.: generate_ds_from_img_dir([funsd_plus_val,funsd_plus_train,funsd_plus_test, cord_train,
        # cord_test, cord_dev, sorie_train, sorie_test])
def generate_ds_from_img_dir(dirs, save_to_path):
    # 1 get file_lists
    all_files = []
    for dir in dirs:
        files = get_imgs_list(dir)
        all_files += files
    print('total files:',len(all_files))
    random.Random(88).shuffle(all_files)    # shuffle

    # 2 generate dataset for file_list
    mydataset = tesseract4img.imgs_to_dataset_generator(all_files)


    print(mydataset)
    # 3 save dataset
    saveto = '/home/ubuntu/air/vrdu/datasets/rvl_HF_datasets/funsd_cord_sorie_dataset.hf'
    mydataset.save_to_disk(saveto)


def generate_rvlcdip_ds():
    dir = '/home/ubuntu/air/vrdu/datasets/images'

    for split in ['val','test']:
        files,labels = get_img_label_pairs(split)
        print(split, ' to be generated file num:',len(files))

        if split in ['train', 'test', 'val']:
            file_part5, label_part5 = _split(files,5), _split(labels,5)

            for i, (sub_files, sub_labels) in enumerate(zip(file_part5, label_part5)):
                print(split, i, ' to be generated file num:',len(sub_files))

                mydataset = tesseract4img.imgs_to_dataset_generator(sub_files,sub_labels)
                saveto = '/home/ubuntu/air/vrdu/datasets/rvl_HF_datasets/full_rvl_'+split+str(i)+'_dataset.hf'
                mydataset.save_to_disk(saveto)
                print(mydataset)
        else:
            mydataset = tesseract4img.imgs_to_dataset_generator(files,labels)
            saveto = '/home/ubuntu/air/vrdu/datasets/rvl_HF_datasets/full_rvl_'+split+'_dataset.hf'
            mydataset.save_to_disk(saveto)

            print(mydataset)


def generate_cdip_ds(dir, all_imgs=None):
    if not all_imgs:
        all_imgs = get_imgs_dfs(dir, '.tif')
    # print(len(res))
    # print(res[:5])
    random.Random(88).shuffle(all_imgs)

    split_imgs = _split(all_imgs,20)

    for i,sub_imgs in enumerate(split_imgs):
        if i==0: continue
        print(i, ' to be generated:', len(sub_imgs))
        mydataset = tesseract4img.imgs_to_dataset_generator(sub_imgs)

        saveto = '/home/ubuntu/air/vrdu/datasets/rvl_HF_datasets/full_cdip_b'+str(i)+'_dataset.hf'
        mydataset.save_to_disk(saveto)
        print(mydataset)


# if __name__ == '__main__':
#     # get all predicts
#     img_paths, labels = util.read_pairs('')
#     # get sampled subset
#     sub_imgs, sub_labels = get_ratioly_sampled(img_paths, labels)

#     for img,lab in zip(sub_imgs, sub_labels):
#         util.write_line('sampled_a.txt',img+'\t'+lab)


# if __name__ == '__main__':
#     # load datasets, with concatenation; 
#     d1 = '/home/ubuntu/air/vrdu/datasets/rvl_HF_datasets/full_cdip_a0_dataset.hf'
#     d1 = load_from_disk(d1)
#     d2 = '/home/ubuntu/air/vrdu/datasets/rvl_HF_datasets/full_cdip_a1_dataset.hf'
#     d2 = load_from_disk(d2)
#     d3 = '/home/ubuntu/air/vrdu/datasets/rvl_HF_datasets/full_cdip_a2_dataset.hf'
#     d3 = load_from_disk(d3)
#     all_ds = concatenate_datasets([d1,d2,d3])

#     # load sampled set,
#     img_paths, labels = util.read_pairs('')
#     lookup = set(img_paths)
#     print(len(lookup))

#     # filter with the set
#     ds = all_ds.filter(lambda sample: sample['image'] in lookup, num_proc=os.cpu_count())
#     print(ds)

#     ds.save_to_disk('temp_a.hf')

# if __name__ == '__main__':
#     dir = '/home/ubuntu/air/vrdu/datasets/cdip_v1/imagesd'
#     all_imgs = get_imgs_dfs(dir, '.tif')
#     for img in all_imgs:
#         # print(img)
#         util.write_line('d_imgs.txt', img)

if __name__ == '__main__':
    # dir = '/home/ubuntu/air/vrdu/datasets/cdip_v1/imagesb/'
    all_img_paths = '/home/ubuntu/air/vrdu/datasets/cdip_v1/b_imgs.txt'
    all_imgs = util.read_lines(all_img_paths)
    # print('iterate dir:', dir)
    generate_cdip_ds(dir, all_imgs) 

