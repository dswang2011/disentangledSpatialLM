import os
import glob
import json 
import random
from pathlib import Path
from difflib import SequenceMatcher


# import cv2
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from IPython.display import display
import matplotlib
from matplotlib import pyplot, patches


# load boxes from box files
def read_bbox_and_words(path: Path):
    bbox_and_words_list = []

    with open(path, 'r', errors='ignore') as f:
        for line in f.read().splitlines():
            if len(line) == 0:
                continue
        
            split_lines = line.split(",")

            bbox = np.array(split_lines[0:8], dtype=np.int32)
            text = ",".join(split_lines[8:])

            # From the splited line we save (filename, [bounding box points], text line).
            # The filename will be useful in the future
            bbox_and_words_list.append([path.stem, *bbox, text])
    
    dataframe = pd.DataFrame(bbox_and_words_list, 
        columns=['filename', 'x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'line'])
    dataframe = dataframe.drop(columns=['x1', 'y1', 'x3', 'y3'])
    # keep: filename	x0	y0	x2	y2	line
    # e.g.: 	X51005365187	17	35	371	91	3-1707067
    return dataframe


# load entities from entity files, entities/xxx
def read_entities(path: Path):
    with open(path, 'r') as f:
        data = json.load(f)
    dataframe = pd.DataFrame([data])
    return dataframe

'''
{
    "company": "F&P PHARMACY",
    "date": "02/03/2018",
    "address": "NO.20. GROUND FLOOR, JALAN BS 10/6 TAMAN BUKIT SERDANG, SEKSYEN 10, 43300 SERI KEMBANGAN. SELANGOR DARUL EHSAN",
    "total": "31.90"
}
'''


# Assign a label to the line by checking the similarity
# of the line and all the entities
def assign_line_label(line: str, entities: pd.DataFrame):
    line_set = line.replace(",", "").strip().split()
    for i, column in enumerate(entities):
        entity_values = entities.iloc[0, i].replace(",", "").strip()
        entity_set = entity_values.split()
         
        matches_count = 0
        for l in line_set:
            if any(SequenceMatcher(a=l, b=b).ratio() > 0.8 for b in entity_set):
                matches_count += 1
            
            if (column.upper() == 'ADDRESS' and (matches_count / len(line_set)) >= 0.5) or \
               (column.upper() != 'ADDRESS' and (matches_count == len(line_set))) or \
               matches_count == len(entity_set):
                return column.upper()
    return "O"



def assign_labels(words: pd.DataFrame, entities: pd.DataFrame):
    max_area = {"TOTAL": (0, -1), "DATE": (0, -1)}  # Value, index
    already_labeled = {"TOTAL": False,
                       "DATE": False,
                       "ADDRESS": False,
                       "COMPANY": False,
                       "O": False
    }
    # Go through every line in $words and assign it a label
    labels = []
    for i, line in enumerate(words['line']):
        label = assign_line_label(line, entities)

        already_labeled[label] = True
        if (label == "ADDRESS" and already_labeled["TOTAL"]) or \
           (label == "COMPANY" and (already_labeled["DATE"] or already_labeled["TOTAL"])):
            label = "O"

        # Assign to the largest bounding box
        if label in ["TOTAL", "DATE"]:
            x0_loc = words.columns.get_loc("x0")
            bbox = words.iloc[i, x0_loc:x0_loc+4].to_list()
            area = (bbox[2] - bbox[0]) + (bbox[3] - bbox[1])

            if max_area[label][0] < area:
                max_area[label] = (area, i)

            label = "O"

        labels.append(label)

    labels[max_area["DATE"][1]] = "DATE"
    labels[max_area["TOTAL"][1]] = "TOTAL"

    words["label"] = labels
    return words
'''
result in: 
schema: filename	x0	y0	x2	y2	line	label
e.g.: X51005365187	17	35	371	91	3-1707067	O
'''


# split the line in to words
# def split_line(line: pd.Series):
#     line_copy = line.copy()
#     line_str = line_copy.loc["line"]
#     words = line_str.split(" ")

#     # Filter unwanted tokens
#     words = [word for word in words if len(word) >= 1]

#     x0, y0, x2, y2 = line_copy.loc[['x0', 'y0', 'x2', 'y2']]
#     bbox_width = x2 - x0
  
#     new_lines = []
#     for index, word in enumerate(words):
#         x2 = x0 + int(bbox_width * len(word)/len(line_str))
#         line_copy.at['x0']=x0
#         line_copy.at['x2']=x2
#         line_copy.at['line']=word
        
#         new_lines.append(line_copy.to_list())
#         x0 = x2 + 5 

#     return new_lines

def split_line(line: pd.Series):
    line_copy = line.copy()
    line_str = line_copy.loc["line"]
    words = line_str.split(" ")

    # Filter unwanted tokens
    words = [word for word in words if len(word) >= 1]

    # x0, y0, x2, y2 = line_copy.loc[['x0', 'y0', 'x2', 'y2']]
    # bbox_width = x2 - x0
  
    new_lines = []
    for index, word in enumerate(words):
        # x2 = x0 + int(bbox_width * len(word)/len(line_str))
        # line_copy.at['x0']=x0
        # line_copy.at['x2']=x2
        line_copy.at['line']=word
        
        new_lines.append(line_copy.to_list())
        # x0 = x2 + 5 

    return new_lines

def zip_four_lists(a, b, c, d):
    result = [[a[i], b[i], c[i], d[i]] for i in range(min(len(a), len(b), len(c), len(d)))]
    return result

from time import perf_counter
def dataset_creator(folder: Path):
    bbox_folder = folder / 'box'
    entities_folder = folder / 'entities'
    img_folder = folder / 'img'

    # Sort by filename so that when zipping them together
    # we don't get some other file (just in case)
    entities_files = sorted(entities_folder.glob("*.txt"))
    bbox_files = sorted(bbox_folder.glob("*.txt"))
    img_files = sorted(img_folder.glob("*.jpg"))

    data = []
    
    # print("Reading dataset:")
    for bbox_file, entities_file, img_file in \
            tqdm(zip(bbox_files, entities_files, img_files), total=len(bbox_files)):            
        # Read the files
        bbox = read_bbox_and_words(bbox_file)
        entities = read_entities(entities_file)
        image = Image.open(img_file).convert("RGB")

        # Assign labels to lines in bbox using entities
        bbox_labeled = assign_labels(bbox, entities)
        del bbox


        block_idx = 1
        block_ids = []
        # Split lines into separate tokens
        new_bbox_l = []
        for index, row in bbox_labeled.iterrows():
            token_rows = split_line(row)
            new_bbox_l += split_line(row)
            block_ids += [block_idx] * len(token_rows)
            block_idx+=1
        new_bbox = pd.DataFrame(new_bbox_l, columns=bbox_labeled.columns)
        del bbox_labeled

        
        # Do another label assignment to keep the labeling more precise 
        for index, row in new_bbox.iterrows():
            label = row['label']

            if label != "O":
                entity_values = entities.iloc[0, entities.columns.get_loc(label.lower())]
                entity_set = entity_values.split()
            
                if any(SequenceMatcher(a=row['line'], b=b).ratio() > 0.7 for b in entity_set):
                    label = "S-" + label
                else:
                    label = "O"
        
            new_bbox.at[index, 'label'] = label

        width, height = image.size

        # print('=start=',new_bbox, ' 222 ',width, height,'=end=')
        # print(img_file)
        # print('-----')
        data.append([new_bbox, width, height])

        bboxes = zip_four_lists(new_bbox['x0'],new_bbox['y0'],new_bbox['x2'],new_bbox['y2'])
        yield {
            'image_path':str(img_file), 'tokens':new_bbox['line'], 'labels': new_bbox['label'], 
            'bboxes':bboxes, 'block_ids': block_ids
        }

    return data

def normalize(points: list, width: int, height: int) -> list:
    x0, y0, x2, y2 = [int(p) for p in points]

    x0 = int(1000 * (x0 / width))
    x2 = int(1000 * (x2 / width))
    y0 = int(1000 * (y0 / height))
    y2 = int(1000 * (y2 / height))

    return [x0, y0, x2, y2]


def write_dataset(dataset: list, output_dir: Path, name: str):
    print(f"Writing {name}ing dataset:")
    with open(output_dir / f"{name}.txt", "w+", encoding="utf8") as file, \
        open(output_dir / f"{name}_box.txt", "w+", encoding="utf8") as file_bbox, \
        open(output_dir / f"{name}_image.txt", "w+", encoding="utf8") as file_image:

        # Go through each dataset
        for datas in tqdm(dataset, total=len(dataset)):
            data, width, height = datas

            filename = data.iloc[0, data.columns.get_loc('filename')]
  
            # Go through every row in dataset
            for index, row in data.iterrows():
                bbox = [int(p) for p in row[['x0', 'y0', 'x2', 'y2']]]
                normalized_bbox = normalize(bbox, width, height)

                file.write("{}\t{}\n".format(row['line'], row['label']))
                file_bbox.write("{}\t{} {} {} {}\n".format(row['line'], *normalized_bbox))
                file_image.write("{}\t{} {} {} {}\t{} {}\t{}\n".format(row['line'], *bbox, width, height, filename))

        # Write a second newline to separate dataset from others
        file.write("\n")
        file_bbox.write("\n")
        file_image.write("\n")




sroie_folder_path = Path('/home/ubuntu/air/vrdu/datasets/sorie2019')
example_file = Path('X51005365187.txt')

# Example usage
# entities_file_path = sroie_folder_path /  "test/entities" / example_file
# bbox_file_path = sroie_folder_path / "test/box" / example_file
# bbox = read_bbox_and_words(path=bbox_file_path)
# entities = read_entities(path=entities_file_path)
# bbox_labeled = assign_labels(bbox, entities)
# res = bbox_labeled.head(15)
# print(res)
# # Example usage
# new_lines = split_line(bbox_labeled.loc[1])
# print("Original row:")
# display(bbox_labeled.loc[1:1,:])

# print("Splitted row:")
# pd.DataFrame(new_lines, columns=bbox_labeled.columns)

# dataset_train = dataset_creator(sroie_folder_path / 'train')
# dataset_test = dataset_creator(sroie_folder_path / 'test')
from datasets import Dataset
ds_test = Dataset.from_generator(dataset_creator, gen_kwargs={'folder':sroie_folder_path / 'test'})
ds_train = Dataset.from_generator(dataset_creator, gen_kwargs={'folder':sroie_folder_path / 'train'})
ds_test.save_to_disk("sorie_test.hf")
ds_train.save_to_disk("sorie_train.hf")

print(ds_train)
print(ds_test)



# dataset_directory = Path('temp_dataset','dataset')

# dataset_directory.mkdir(parents=True, exist_ok=True)

# write_dataset(dataset_train, dataset_directory, 'train')
# write_dataset(dataset_test, dataset_directory, 'test')

# # Creating the 'labels.txt' file to the the model what categories to predict.
# labels = ['COMPANY', 'DATE', 'ADDRESS', 'TOTAL']
# IOB_tags = ['S']
# with open(dataset_directory / 'labels.txt', 'w') as f:
#     for tag in IOB_tags:
#         for label in labels:
#             f.write(f"{tag}-{label}\n")
#     # Writes in the last label O - meant for all non labeled words
#     f.write("O")
