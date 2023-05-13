import torch
import numpy as np
import os
from torch.utils.data import DataLoader
import json

class MyInferencer:
    def __init__(self,opt):
        self.opt = opt


    def inference_for_classification(self, opt, model, mydata):
        trainable_ds = mydata.trainable_ds
        raw_ds = mydata.raw_ds
        train_dataloader = DataLoader(trainable_ds, batch_size=opt.batch_size)
        print('-- start to infer --')
        all_preds = []
        img_paths = []
        model.eval()
        with torch.no_grad():
            for batch_index, inputs in enumerate(train_dataloader):
                # move to the device
                inputs = {key: value.to(opt.device) for key, value in inputs.items()}
                # infer
                outputs = model(**inputs)
                # get label id
                batch_predictions = torch.argmax(outputs.logits, dim=-1).tolist()
                batch_imgs = [raw_ds[i + opt.batch_size*batch_index]['image'] for i in range(len(batch_predictions))]
                all_preds.extend(batch_predictions)
                img_paths.extend(batch_imgs)
        return img_paths,all_preds


    def inference_for_QA(self, model, mydata, save_to_file):
        # 1 load dataset
        # test_dataset = mydata.test_dataset
        loader_test = DataLoader(mydata.test_ds, batch_size=self.opt.batch_size*3)

        model.eval()
        res = []
        # 2. iterate and inference
        # for row in test_dataset:
        for batch_index, batch in enumerate(loader_test):
            print(batch_index)
            questionId = batch['questionId']
            input_ids = batch['input_ids'].to(self.opt.device)
            position_ids = batch['position_ids'].to(self.opt.device)
            attention_mask = batch['attention_mask'].to(self.opt.device)
            token_type_ids = batch['token_type_ids'].to(self.opt.device)
            bbox = batch['bbox'].to(self.opt.device)
            # start_positions = row['ans_start_positions'].to(opt.device)
            # end_positions = row['ans_end_positions'].to(opt.device)
            pixel_values = batch['pixel_values'].to(self.opt.device)

            with torch.no_grad():
                outputs = model(input_ids = input_ids, position_ids=position_ids,
                        token_type_ids=token_type_ids, bbox = bbox, attention_mask = attention_mask, 
                        pixel_values = pixel_values)
                # 3.1 get start_logits and end_logits
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits

                # 3.2: get largest logit for both
                for idx, (s,e) in enumerate(zip(start_logits,end_logits)):
                    predicted_start_idx = s.argmax(-1).item()    # item() is to get scalars
                    predicted_end_idx = e.argmax(-1).item()
                    # print("Predicted start idx:", predicted_start_idx)
                    # print("Predicted end idx:", predicted_end_idx)
                    # 4. get the text answer
                    answer = mydata.tokenizer.decode(input_ids[idx][predicted_start_idx:predicted_end_idx+1]) 
                    res.append({"questionId":questionId[idx].item(), "answer":answer})
        # save it
        res = json.dumps(res)
        self.save_res(save_to_file,res)
        return res

    def save_res(self,path,performance_str):
        with open(path, 'w') as f:
            f.write(str(performance_str))

