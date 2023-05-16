from datasets import load_dataset ,Features, Sequence, Value, Array2D, Array3D, Dataset, DatasetDict
from datasets.features import ClassLabel
from transformers import AutoProcessor, AutoTokenizer, AutoConfig
import os
import json
import transformers
from PIL import Image
from mydataset import myds_util

class CORD:
    def __init__(self,opt) -> None:    
        self.opt = opt
        '''
        DatasetDict({
            train: Dataset({
                features: ['id', 'words', 'bboxes', 'ner_tags', 'image'],
                num_rows: 800
            })
        })
        '''
        # step1: create config, tokenizer, and processor
        self.config = AutoConfig.from_pretrained(opt.layoutlm_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(opt.layoutlm_dir)
        assert isinstance(self.tokenizer, transformers.PreTrainedTokenizerFast)  # get sub
        self.processor = AutoProcessor.from_pretrained(opt.layoutlm_dir,tokenizer=self.tokenizer, apply_ocr=False)    # wrap of featureExtract & tokenizer

        self.label_col_name = "ner_tags"

        # step 2.1: get raw ds (already normalized bbox, img object)
        raw_train, raw_test = self.get_raw_ds()

        raw_train, raw_test = self.get_img_and_norm_ds(raw_train), self.get_img_and_norm_ds(raw_test)

        # step 2.2, get  labeled ds (get label list, and map label dataset)
        _,_, opt.label_list = self._get_label_map(raw_train)
        opt.num_labels = len(opt.label_list)    # 54; 27 pairs
        self.class_label = ClassLabel(num_classes=opt.num_labels, names = opt.label_list)

        # map label ds
        train_label_ds, test_label_ds = self.get_label_ds(raw_train), self.get_label_ds(raw_test)

        # step 3: prepare for getting trainable data (encode and define features)
        
        trainable_train_ds, trainable_test_ds = self.get_preprocessed_ds(train_label_ds),self.get_preprocessed_ds(test_label_ds)
        
        print(trainable_train_ds)
        # print(trainable_train_ds[0])
        self.trainable_ds = DatasetDict({
            "train" : trainable_train_ds , 
            "test" : trainable_test_ds 
        })


    def get_raw_ds(self):
        train = Dataset.from_generator(self.ds_generator, gen_kwargs={'base_dir':self.opt.cord_train})
        test = Dataset.from_generator(self.ds_generator, gen_kwargs={'base_dir':self.opt.cord_test})
        return train, test

    def ds_generator(self, base_dir):
        # logger.info("⏳ Generating examples from = %s", filepath)
        ann_dir = os.path.join(base_dir, "json")
        img_dir = os.path.join(base_dir, "image")

        block_idx = 1
        for doc_idx, file in enumerate(sorted(os.listdir(ann_dir))):
            # print('---doc id:---',doc_idx)
            tokens = []
            bboxes = []
            tboxes = []
            ner_tags = []
            block_ids = []

            file_path = os.path.join(ann_dir, file)
            with open(file_path, "r", encoding="utf8") as f:
                data = json.load(f)
            image_path = os.path.join(img_dir, file)
            image_path = image_path.replace("json", "png")
            image, size = self._load_image(image_path)
            block_idx = 1
            for block in data["valid_line"]:
                cur_line_bboxes = []
                line_words, label = block["words"], block["category"]
                words = [w for w in line_words if w["text"].strip() != ""]
                if len(words) == 0:
                    continue
                if label == "other":
                    for w in words:
                        tokens.append(w["text"])
                        block_ids.append(block_idx)
                        ner_tags.append("O")
                        t_box = self._quad_to_box(w["quad"])
                        tboxes.append(t_box)
                        cur_line_bboxes.append(t_box)
                else:
                    tokens.append(words[0]["text"])
                    block_ids.append(block_idx)
                    ner_tags.append("B-" + label.upper())
                    t_box = self._quad_to_box(words[0]["quad"])
                    tboxes.append(t_box)
                    cur_line_bboxes.append(t_box)
                    for w in words[1:]:
                        tokens.append(w["text"])
                        block_ids.append(block_idx)
                        ner_tags.append("I-" + label.upper())
                        t_box = self._quad_to_box(w["quad"])
                        tboxes.append(t_box)
                        cur_line_bboxes.append(t_box)
                cur_line_bboxes = self._get_line_bbox(cur_line_bboxes)  # shared boxes, token-wise
                bboxes.extend(cur_line_bboxes)
                block_idx += 1

            yield {"id": doc_idx, "tokens": tokens, "bboxes": bboxes,"tboxes": tboxes, "ner_tags": ner_tags,
                   "block_ids": block_ids, "image": image_path}
        # one_page_info = {'tokens': [], 'tboxes': [], 'bboxes': [], 'block_ids':[], 'image': image_path}

    def get_img_and_norm_ds(self,ds):
        def _load_and_norm(sample):
            # 1) load img obj
            sample['images'],size = self._load_image(sample['image'])
            # 2) normalize bboxes using the img size 
            sample['bboxes'] = [self._normalize_bbox(bbox, size) for bbox in sample['bboxes']]
            return sample
        normed_ds = ds.map(_load_and_norm, num_proc=os.cpu_count()) # load image objects
        return normed_ds


    def get_preprocessed_ds(self,ds):
        def _preprocess(batch):
            # 1) encode words and imgs
            encodings = self.processor(images=batch['images'],text=batch['tokens'], boxes=batch['bboxes'],
                                       word_labels=batch['ner_tags'], truncation=True, padding='max_length', max_length=self.opt.max_seq_len)
            # 2) add position_ids
            position_ids = []
            for i, block_ids in enumerate(batch['block_ids']):
                word_ids = encodings.word_ids(i)
                rel_pos = self._get_rel_pos(word_ids, block_ids)
                position_ids.append(rel_pos)
            encodings['position_ids'] = position_ids
            return encodings

        features = Features({
            'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
            'input_ids': Sequence(feature=Value(dtype='int64')),
            'position_ids': Sequence(feature=Value(dtype='int64')),
            'attention_mask': Sequence(Value(dtype='int64')),
            'bbox': Array2D(dtype="int64", shape=(512, 4)),
            'labels': Sequence(feature=Value(dtype='int64')),
        })
        # processed_ds = ds.map(_preprocess, batched=True, num_proc=self.cpu_num, 
        #     remove_columns=['id','tokens', 'bboxes','ner_tags','block_ids','image'], features=features).with_format("torch")
        processed_ds = ds.map(_preprocess, batched=True, num_proc=os.cpu_count(), 
            remove_columns=ds.column_names, features=features,batch_size=100).with_format("torch")
    
        # process to: 'input_ids', 'position_ids','attention_mask', 'bbox', 'labels', 'pixel_values']
        return processed_ds


    def get_label_ds(self,ds):
        def map_label2id(sample):
            sample['ner_tags'] = [self.class_label.str2int(ner_label) for ner_label in sample['ner_tags']]
            return sample
        label_ds = ds.map(map_label2id, num_proc=os.cpu_count())
        return label_ds


    def _get_label_list(self,labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list
    # get label list
    def _get_label_map(self,ds):
        features = ds.features
        column_names = ds.column_names
        if isinstance(features[self.label_col_name].feature, ClassLabel):
            label_list = features[self.label_col_name].feature.names
            # No need to convert the labels since they are already ints.
            id2label = {k: v for k,v in enumerate(label_list)}
            label2id = {v: k for k,v in enumerate(label_list)}
        else:
            label_list = self._get_label_list(ds[self.label_col_name])   # this invokes another function
            id2label = {k: v for k,v in enumerate(label_list)}
            label2id = {v: k for k,v in enumerate(label_list)}
        return id2label, label2id, label_list

    def _load_image(self,image_path):
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        return image, (w, h)

    def _normalize_bbox(self,bbox, size):
        return [
            int(1000 * bbox[0] / size[0]),
            int(1000 * bbox[1] / size[1]),
            int(1000 * bbox[2] / size[0]),
            int(1000 * bbox[3] / size[1]),
        ]
    def _get_line_bbox(self,bboxs):
        x = [bboxs[i][j] for i in range(len(bboxs)) for j in range(0, len(bboxs[i]), 2)]
        y = [bboxs[i][j] for i in range(len(bboxs)) for j in range(1, len(bboxs[i]), 2)]
        x0, y0, x1, y1 = min(x), min(y), max(x), max(y)
        assert x1 >= x0 and y1 >= y0
        bbox = [[x0, y0, x1, y1] for _ in range(len(bboxs))]
        return bbox
    
    def _quad_to_box(self, quad):
    # test 87 is wrongly annotated
        box = (
            max(0, quad["x1"]),
            max(0, quad["y1"]),
            quad["x3"],
            quad["y3"]
        )
        if box[3] < box[1]:
            bbox = list(box)
            tmp = bbox[3]
            bbox[3] = bbox[1]
            bbox[1] = tmp
            box = tuple(bbox)
        if box[2] < box[0]:
            bbox = list(box)
            tmp = bbox[2]
            bbox[2] = bbox[0]
            bbox[0] = tmp
            box = tuple(bbox)
        return box

    def _get_rel_pos(self,word_ids, block_ids):   # [None, 0, 1, 2, 2, 3, None]; [1,1,2,2] which is dict {word_idx: block_num}
        res = []
        rel_cnt = self.config.pad_token_id+1
        prev_block = 1
        for word_id in word_ids:
            if word_id is None:
                res.append(self.config.pad_token_id)
                continue
            else:
                curr_block = block_ids[word_id]   # word_id is the 0,1,2,3,.. word index;
                if curr_block != prev_block:
                    # set back to 0; 
                    rel_cnt = self.config.pad_token_id+1
                    res.append(rel_cnt) # operate
                    # reset prev_block
                    prev_block = curr_block
                else:
                    res.append(rel_cnt)
            rel_cnt+=1
        return res

if __name__ == '__main__':
    # Section 1, parse parameters
    mydata = FUNSD(None)
    test_dataset = mydata.get_data(split='test')
    print(test_dataset)
    doc1 = test_dataset[0]
    print(doc1['input_ids'])

    '''
    Dataset({
        features: ['input_ids', 'attention_mask', 'bbox', 'labels', 'pixel_values'],
        num_rows: 100
    })
    '''

