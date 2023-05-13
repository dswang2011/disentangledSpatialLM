from PIL import Image
import pytesseract
import pickle
import os
import json
from datasets import Dataset, load_from_disk
import numpy as np
from OCRs import img_util
# import tok_util
from datasets import Dataset, Features, Sequence, Value, Array2D, Array3D


# skip those that have multiple pages
def _load_image(image_path, convert=False):
    try:
        image = Image.open(image_path)
        num_img = image.n_frames
        if num_img>1:
            print('multiple page, skip')
            return None, (-1,-1)
            # image.seek(0)
        if convert: 
            image = image.convert("RGB")
    except Exception as e:
        print(e)
        return None, (-1,-1)
    w, h = image.size
    return image, (w, h)


def _convert_and_save(img_obj, img_path):
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
        return False


# double checked
def doc_to_segs(one_doc):
    texts, bboxes = [], []
    word_nums = []

    seg_ids = one_doc['seg_ids']
    tokens = one_doc['tokens']
    boxes = one_doc['share_bboxes']  # shared boxes, share_bboxes, bboxes; here, it must be shared because it is seg oriented

    block_num = seg_ids[0]  # 11
    window_tokens = [tokens[0]]
    l = 0
    for i in range(1, len(seg_ids)):
        curr_id = seg_ids[i]
        if curr_id != block_num:
            word_nums.append(len(window_tokens))
            text = ' '.join(window_tokens)
            texts.append(text)
            bboxes.append(boxes[l])
            # reset the params
            l = i
            block_num = curr_id
            window_tokens = [tokens[i]]
        else:
            window_tokens.append(tokens[i])
    word_nums.append(len(window_tokens))
    text = ' '.join(window_tokens)
    texts.append(text)
    bboxes.append(boxes[l])

    return texts, bboxes, word_nums

# prepare dataset dict 1.2: images to dataset
def imgs_to_dataset_generator(img_paths, labels=None, tesseract_wait=False, **kwargs):
    dataset = Dataset.from_generator(image_to_dict, gen_kwargs={'img_paths': img_paths, 'labels':labels, **kwargs})
    return dataset


# prepare dataset dict 1.1: image to basic dict info
def image_to_dict(img_paths, labels =None, tbox_norm=False,tesseract_wait=False,**other_params):
    print('wait?', tesseract_wait)
    '''
    rtype: return one_doc, where the bbox and h/w are normalized to 1000*1000
    '''
    for idx, image_path in enumerate(img_paths):
        one_page_info = {'tokens': [], 'tboxes': [], 'bboxes': [], 'block_ids':[], 'image': image_path}
        if labels:
            one_page_info['label'] = labels[idx]

        # append other params
        for key,val in other_params.items():
            one_page_info[key] = val[idx]

        image, size = _load_image(image_path, convert=False)    # for OCR you dont convert, for model features, you convert;
        if not image or size[0]<=0: continue

        one_page_info['size'] = size

        try:
            myconfig = r'--psm 11 --oem 3'
            if tesseract_wait:
                data = pytesseract.image_to_data(image, config=myconfig, output_type='dict', timeout=30) # 2/3s
            else:
                data = pytesseract.image_to_data(image, config=myconfig, output_type='dict', timeout=20) # 2/3s
        except RuntimeError as timeout_error:
            print(timeout_error)
            print('img:', image_path)
            continue
        except:
            print("Something else went wrong")
            print('img:', image_path)
            continue

        # print(data)
        confs = data['conf']
        texts = data['text']
        page_nums = data['page_num']
        block_nums = data['block_num']
        line_nums = data['line_num']
        x0s = data['left']
        y0s = data['top']
        hs = data['height']  # temporary use only,
        ws = data['width']  # temporary use only

        # encoding = img_util.feature_extractor(image)
        # one_doc['image'] = encoding.pixel_values[0]    # image object, get the first one, cause there is only one!
        # one_doc['image'] = image
        for i, token in enumerate(texts):
            # token
            token = token.strip()
            if token == '': continue
            # if confs[i]<1: continue
            # height and width
            height, width = hs[i], ws[i]
            # coordinate
            x0 = x0s[i]
            y0 = y0s[i]
            x1 = x0 + width
            y1 = y0 + height
            # page, line, block, block_id
            page_num, line_num, block_num = page_nums[i], line_nums[i], block_nums[i]

            # produce one sample
            one_page_info['tokens'].append(token)
            #
            if tbox_norm:
                one_page_info['tboxes'].append(img_util._normalize_bbox([x0, y0, x1, y1], size))
            else:
                one_page_info['tboxes'].append([x0, y0, x1, y1])
            one_page_info['block_ids'].append(block_num)

        # if the page is empty skip
        if not one_page_info['tokens']: continue
        # extend with bboxes, i.e., the shared box
        one_page_info = img_util._extend_shared_bbox(one_page_info)
        if idx%100==0:
            print(idx, one_page_info)
        
        yield one_page_info


def get_img2doc_data(img_dir):
    res = {}  # a dict of dict, i.e., {docID_pageNO : {one_doc_info}}
    for doc_idx, file in enumerate(sorted(os.listdir(img_dir))):
        image_path = os.path.join(img_dir, file)
        one_doc = image_to_doc(image_path)
        docID_pageNO = file.replace(".png", "")
        res[docID_pageNO] = one_doc
        # print(one_doc)
        # if doc_idx>50:
        #     break
    return res


def get_question_pairs(base, split='val'):
    # from json of questions and answers
    file_path = os.path.join(base, split + '_v1.0.json')

    with open(file_path) as fr:
        data = json.load(fr)
    id2trip = {}
    for sample in data['data']:
        qID = sample['questionId']  # numeric e.g., 8366
        question = sample['question']
        # for test set, there is no answersr
        answers = []
        if 'answers' in sample.keys():
            answers = sample['answers']

        ucsf_doc_id = sample['ucsf_document_id']  # e.g.,: txpp0227
        ucsf_doc_page = sample['ucsf_document_page_no']  # e.g.,: 10
        docID_page = ucsf_doc_id + '_' + ucsf_doc_page
        trip_object = (docID_page, question, answers)
        id2trip[qID] = trip_object
    return id2trip


def wrap_and_save(base, split):
    # mydataset = Dataset.from_generator(generator_based_on_questions,gen_kwargs={'split':split, 'base':base})
    id2queryinfo, id2doc = produce_based_on_questions(base, split)
    print('q num:', len(id2queryinfo.keys()))
    print('doc num:', len(id2doc.keys()))
    output_to_pickle([id2queryinfo, id2doc], split + '.pickle')
    # save to disk


from PIL import Image, ImageDraw, ImageFont
if __name__ == '__main__':
    # step1: using OCR to extract seg texts and boxes from img
    img_path = "/Users/dongshengwang/python_projects/Spatial-LM/data/img.png"
    one_page_dict = image_to_dict([img_path])
    # print(dict(one_page_dict))
    # one_page_info = {'tokens': [], 'tboxes': [], 'bboxes': [], 'block_ids': [], 'image': image_path}

    image = Image.open(img_path)
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")

    target_width = 50
    size = 20  # initial guess
    # font = ImageFont.load_default()
    # font = ImageFont.truetype("arial.ttf", size=size)

    for i,tbox in enumerate(one_page_dict['tboxes']):
        token_txt = one_page_dict['tokens'][i]
        bbox = one_page_dict['bboxes'][i]
        # draw.rectangle(tbox, outline='orange', width=2)
        draw.rectangle(bbox, outline='blue', width=3)

        print(tbox)
        print(token_txt)
        # if i>80: break
        # text_length = font.getlength(token_txt)
        # size = int(size * (target_width / text_length))  # update guess
        # font = ImageFont.truetype("arial.ttf", size=size)
        # draw.text((tbox[0], tbox[1]), token_txt, fill='orange', font=font)
    Image._show(image)
    # for annotation in data['form']:
    #   label = annotation['label']
    #   general_box = annotation['box']
    #   draw.rectangle(general_box, outline='blue', width=2)
    #   draw.text((general_box[0] + 10, general_box[1] - 10), label, fill=label2color[label], font=font)
    #   words = annotation['words']
    #   for word in words:
    #     box = word['box']
    #     draw.rectangle(box, outline='orange', width=1)
