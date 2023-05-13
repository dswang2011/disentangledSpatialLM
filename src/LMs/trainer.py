from tqdm import tqdm 
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score,roc_auc_score,classification_report
import time
import numpy as np
import os
import pickle
from datetime import datetime
from accelerate import Accelerator
import evaluate

from datasets import load_metric
metric = evaluate.load("seqeval")



def finetune(opt,model, mydata):
    # 1 data loader
    loader_train = DataLoader(mydata.trainable_dataset['train'], batch_size=opt.batch_size,shuffle=True)
    loader_test = DataLoader(mydata.trainable_dataset['test'], batch_size=opt.batch_size,shuffle=False)
    
    # 2 optimizer
    optimizer = torch.optim.AdamW(model.parameters(),lr=opt.lr, betas=(0.9,0.999),eps=1e-08)
    # 3 training
    best_f1 = 0.0
    best_loss = 99999.9

    for epoch in range(opt.epochs):    
        print('epoch:',epoch,'/',str(opt.epochs))
        # train mode
        model.train()
        for batch in tqdm(loader_train, desc = 'Training'):
        # for idx,batch in enumerate(loader_train):
            optimizer.zero_grad()  # Clear gradients.
            outputs = predict_one_batch(opt,model,batch,eval=False)
            loss = outputs.loss
            # print('Loss:',loss.item())
            loss.backward()
            optimizer.step()  # Update parameters based on gradients.
        
        # eval mode
        if opt.task_type in ['docvqa','cspretrain']:
            # if loss.item()<best_loss:
            best_loss=loss.item()
            save_model(opt, model,{'loss':best_loss})
            print('The best model saved with loss:', best_loss, ' to ', opt.dir_path)
            continue
        # if finetune models
        if opt.task_type == 'token-classifier':
            res_dict = test_eval(opt,model,loader_test)
            if res_dict['f1']>best_f1:
                save_model(opt, model,res_dict)
                best_f1 = res_dict['f1']
                print('The best model saved with f1:', best_f1,' to ', opt.dir_path)
    return best_f1

def pretrain(opt, model, mydata):
    # 1 data loader
    loader_train = DataLoader(mydata.masked_train_dataset, batch_size=opt.batch_size,shuffle=True)
    # 2 optimizer
    optimizer = torch.optim.AdamW(model.parameters(),lr=opt.lr, betas=(0.9,0.999),eps=1e-08)


    if bool(opt.multi_gpu_accelerator):
        accelerator = Accelerator()
        loader_train, model, optimizer = accelerator.prepare(
            loader_train, model, optimizer
        )    
    
    best_f1 = 0.0
    best_loss = 99999.9

    for epoch in range(opt.epochs):    
        print('epoch:',epoch,'/',str(opt.epochs))
        # train mode
        model.train()
        for batch in tqdm(loader_train, desc = 'Training'):
        # for idx,batch in enumerate(loader_train):
            optimizer.zero_grad()  # Clear gradients.
            outputs = predict_one_batch(opt,model,batch,eval=False)
            loss = outputs.loss
            if bool(opt.multi_gpu_accelerator):
                accelerator.backward(loss)
            else:
                loss.backward()
            optimizer.step()  # Update parameters based on gradients.
            # if idx>10: break
        # eval mode
        if opt.task_type == 'cspretrain':
            # if loss.item()<best_loss:
            best_loss=loss.item()
            csmodel = model.cs_layoutlm  # to be saved
            save_pretrain_model(opt, csmodel ,{'loss':best_loss})
            print('The best layoutlm model saved with loss:', best_loss, ' to ', opt.dir_path)

    return best_f1


def test_eval(opt,model,loader_test):
    # test
    model.eval()
    preds,tgts, val_loss = predict_all_batches(opt, model,loader_test)
    print(f'val Loss: {val_loss:.4f}')

    # res_dict = evaluate(preds,tgts)
    if opt.task_type == 'docvqa':
        return {'val_loss':val_loss}
    
    res_dict = compute_metrics(opt, [preds,tgts])
    print(res_dict)

    return res_dict

# for backpropagation use, so define the input variables
def predict_one_batch(opt, model, batch, eval=False):
    if opt.task_type == 'cspretrain':
        if bool(opt.multi_gpu_accelerator):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            dist = batch['dist']
            direct = batch['direct']
            seg_width = batch['seg_width']
            seg_height = batch['seg_height']
            labels = batch['labels']
            segmentation_ids = torch.tensor([[0,1,2,3,4,5,6,7,8] for _ in range(len(input_ids))])
        else:
            input_ids = batch['input_ids'].to(opt.device)
            attention_mask = batch['attention_mask'].to(opt.device)
            dist = batch['dist'].to(opt.device)
            direct = batch['direct'].to(opt.device)
            seg_width = batch['seg_width'].to(opt.device)
            seg_height = batch['seg_height'].to(opt.device)
            labels = batch['labels'].to(opt.device)
            segmentation_ids = torch.tensor([[0,1,2,3,4,5,6,7,8] for _ in range(len(input_ids))])
            segmentation_ids.to(opt.device)

        if eval:
            with torch.no_grad():
                outputs = model(
                    input_ids = input_ids, attention_mask = attention_mask, dist = dist, 
            direct = direct, seg_width=seg_width, seg_height=seg_height, 
            segmentation_ids=None, labels = labels)  
        else:
            outputs = model(
                input_ids = input_ids, attention_mask = attention_mask, dist = dist, 
            direct = direct, seg_width=seg_width, seg_height=seg_height, 
            segmentation_ids=segmentation_ids, labels = labels)

    elif opt.task_type == 'token-classifier':
        input_ids = batch['input_ids'].to(opt.device)
        attention_mask = batch['attention_mask'].to(opt.device)
        dist = batch['dist'].to(opt.device)
        direct = batch['direct'].to(opt.device)
        seg_width = batch['seg_width'].to(opt.device)
        seg_height = batch['seg_height'].to(opt.device)
        labels = batch['labels'].to(opt.device)
        segmentation_ids = torch.tensor([[0,1,2,3,4,5,6,7,8] for _ in range(len(input_ids))])
        segmentation_ids.to(opt.device)

        if eval:
            with torch.no_grad():
                outputs = model(
                    input_ids = input_ids,attention_mask = attention_mask, 
                    dist = dist, direct = direct, 
                    seg_width = seg_width,seg_height=seg_height,
                    segmentation_ids=segmentation_ids,labels=labels )  
        else:
            outputs = model(
                    input_ids = input_ids,attention_mask = attention_mask, 
                    dist = dist, direct = direct, 
                    seg_width = seg_width,seg_height=seg_height,
                    segmentation_ids=segmentation_ids,labels=labels)

    elif opt.task_type == 'docvqa':
        input_ids = batch['input_ids'].to(opt.device)
        attention_mask = batch['attention_mask'].to(opt.device)
        token_type_ids = batch['token_type_ids'].to(opt.device)
        bbox = batch['bbox'].to(opt.device)
        start_positions = batch['ans_start_positions'].to(opt.device)
        end_positions = batch['ans_end_positions'].to(opt.device)
        pixel_values = batch['pixel_values'].to(opt.device)

        if eval:
            with torch.no_grad():
                outputs = model(input_ids = input_ids, 
                    token_type_ids=token_type_ids, bbox = bbox, attention_mask = attention_mask, 
                    pixel_values = pixel_values, start_positions = start_positions, end_positions=end_positions)  
        else:
            outputs = model(input_ids = input_ids, 
                token_type_ids=token_type_ids, bbox = bbox, attention_mask = attention_mask, 
                pixel_values = pixel_values, start_positions = start_positions, end_positions=end_positions)
    return outputs

# for evaluation use (all batche inference); must be used as eval mode; 
def predict_all_batches(opt,model,dataloader):
        preds, tgts, val_loss = [],[], 0.0
        for _ii, batch in enumerate(dataloader, start=0):
            outputs = predict_one_batch(opt,model, batch, eval=True)
            val_loss+=outputs.loss.item()

            if opt.task_type != 'docvqa':
                predictions = torch.argmax(outputs.logits, dim=-1)
                target = batch['labels']
                preds.append(predictions)   # logits
                tgts.append(target)
        if opt.task_type != 'docvqa':
            preds = torch.cat(preds)
            tgts = torch.cat(tgts)
        return preds,tgts,val_loss


# def inference_docvqa(opt,model, dataloader):
#     pass

# preds is the logits (label distribution);
def evaluate(preds, targets, print_confusion=False):
    # n_total,num_classes = outputs.shape
    # 2) move to cpu to convert to numpy
    preds = preds.cpu().numpy()
    target = targets.cpu().numpy()

    # confusion = confusion_matrix(output, target)
    f1 = f1_score(target, preds, average='weighted')
    precision,recall,fscore,support = precision_recall_fscore_support(target, preds, average='weighted')
    acc = accuracy_score(target, preds)
    performance_dict = {'num':len(preds),'acc': round(acc,3), 'f1': round(f1,3), 'precision':round(precision,3),'recall':round(recall,3)}
    if print_confusion: print(classification_report(target, preds))

    return performance_dict

def compute_metrics(opt, p,return_entity_level_metrics=False):
    predictions, labels = p
    # Remove ignored index (special tokens)
    true_predictions = [
        [opt.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [opt.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    if return_entity_level_metrics:
        # Unpack nested dictionaries
        final_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"] = v
            else:
                final_results[key] = value
        return final_results
    else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

def create_save_dir(params):
    if not os.path.exists('tmp_dir'):
        os.mkdir('tmp_dir')

     # Create model dir
    dir_name = '_'.join([params.network_type,params.dataset_name,str(round(time.time()))[-6:]])
    dir_path = os.path.join('tmp_dir', dir_name)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)  

    params.export_to_config(os.path.join(dir_path, 'config.ini'))
    pickle.dump(params, open(os.path.join(dir_path, 'config.pkl'), 'wb'))
    return dir_path

def save_pretrain_model(params, pretrain_model, perform_dict):
    # 1) save the learned model (model and the params used)
    # torch.save(model.state_dict(), os.path.join(params.dir_path, 'model'))

    pretrain_model.save_pretrained(params.dir_path)

    now=datetime.now()
    str_dt = now.strftime("%d/%m/%Y %H:%M:%S")
    perform_dict['finish_time'] = str_dt

    # 2) Write performance string
    eval_path = os.path.join(params.dir_path, 'eval')
    with open(eval_path, 'w') as f:
        f.write(str(perform_dict))

# Save the model
def save_model(params, model, perform_dict):
    # 1) save the learned model (model and the params used)
    torch.save(model.state_dict(), os.path.join(params.dir_path, 'model'))

    now=datetime.now()
    str_dt = now.strftime("%d/%m/%Y %H:%M:%S")
    perform_dict['finish_time'] = str_dt

    perform_dict['dataset_used'] = ''

    # 2) Write performance string
    eval_path = os.path.join(params.dir_path, 'eval')
    with open(eval_path, 'w') as f:
        f.write(str(perform_dict))

