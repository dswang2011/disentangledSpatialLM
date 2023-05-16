from tqdm import tqdm 
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score,roc_auc_score,classification_report

from transformers import TrainingArguments, Trainer
from transformers.data.data_collator import default_data_collator
import numpy as np
from transformers import DataCollatorForLanguageModeling
import evaluate
from LMs.mycollator import BlockMaskingDataCollator


class MyTrainer:
    def __init__(self,opt):
        self.opt = opt

    def pretrain(self,opt, model, mydata):

        # mlm= True uses masked language model; otherwise, causal LM (NTP); 
        # mydata.tokenizer.pad_token = tokenizer.eos_token  # no idea why??
        if opt.task_type == 'mlm':
            data_collator = DataCollatorForLanguageModeling(tokenizer=mydata.tokenizer, mlm=True, mlm_probability=opt.mlm_probability)
        elif opt.task_type == 'blm':
            data_collator = BlockMaskingDataCollator(tokenizer=mydata.tokenizer, mlm=True, mlm_probability=0.035)

        # logging_steps = len(mydata.train_dataset)  //opt.batch_size
        trainable_ds = mydata.trainable_ds.shuffle(seed=88).train_test_split(test_size=opt.test_size)

        training_args = TrainingArguments(
            output_dir = opt.checkpoint_save_path,
            num_train_epochs = opt.epochs,
            learning_rate = opt.lr,
            per_device_train_batch_size = opt.batch_size,
            per_device_eval_batch_size = opt.batch_size,
            weight_decay = 0.01,
            warmup_ratio = 0.001,
            fp16 = True,    # make it train fast
            push_to_hub = False,
            # push_to_hub_model_id = f"layoutlmv3-finetuned-cord"        
            evaluation_strategy = "epoch",
            save_strategy="steps",  # steps, epoch
            overwrite_output_dir=True,  # use only one dir
            prediction_loss_only = True,
            logging_dir='./logs',  
            # log_level = 'info', # ‘debug’, ‘info’, ‘warning’, ‘error’ and ‘critical’, 
            # logging_strategy = 'epoch', # epoch, step, no

            save_steps=2000,
        )
        trainer = Trainer(
            model = model,
            args = training_args,
            train_dataset = trainable_ds['train'],
            eval_dataset = trainable_ds['test'],
            data_collator = data_collator,
            # config = model.config,
            # tokenizer = mydata.tokenizer
        )
        trainer.train()
        trainer.save_model(opt.save_path)


    def train(self,opt, model, mydata):
        # mlm= True uses masked language model; otherwise, causal LM (NTP); 
        # logging_steps = len(mydata.train_dataset)  //opt.batch_size
        trainable_ds = mydata.trainable_ds

        training_args = TrainingArguments(
            output_dir = opt.checkpoint_save_path,
            num_train_epochs = opt.epochs,
            learning_rate = opt.lr,
            per_device_train_batch_size = opt.batch_size,
            per_device_eval_batch_size = opt.batch_size,
            # weight_decay = 0.01,
            # warmup_ratio = 0.05,
            fp16 = True,    # make it train fast
            push_to_hub = False,
            # push_to_hub_model_id = f"layoutlmv3-finetuned-cord"        
            evaluation_strategy = "epoch",
            save_strategy="no",  # no, epoch, steps
            overwrite_output_dir=True,  # use only one dir
            # prediction_loss_only = True,
            logging_dir='./logs',  
            log_level = 'info', # ‘debug’, ‘info’, ‘warning’, ‘error’ and ‘critical’, 
            logging_strategy = 'epoch', # epoch, step, no
            # save_steps=5000,
        )

        trainer = Trainer(
            model = model,
            args = training_args,
            train_dataset = trainable_ds['train'],
            eval_dataset = trainable_ds['test'],
            compute_metrics = self.compute_metrics,
            # data_collator = data_collator,
            # config = model.config,
            # tokenizer = mydata.tokenizer
        )
        trainer.train()
        if bool(self.opt.save_model):
            trainer.save_model(opt.save_path)

    def train_docvqa(self,opt, model, mydata):
        # mlm= True uses masked language model; otherwise, causal LM (NTP); 
        # logging_steps = len(mydata.train_dataset)  //opt.batch_size
        # split into smaller 
        trainable_ds = mydata.trainable_ds.shuffle(seed=88).train_test_split(test_size=0.001)


        training_args = TrainingArguments(
            output_dir = opt.checkpoint_save_path,
            num_train_epochs = opt.epochs,
            learning_rate = opt.lr,
            per_device_train_batch_size = opt.batch_size,
            per_device_eval_batch_size = opt.batch_size,
            # weight_decay = 0.01,
            # warmup_ratio = 0.05,
            fp16 = True,    # make it train fast
            push_to_hub = False,
            # push_to_hub_model_id = f"layoutlmv3-finetuned-cord"        
            evaluation_strategy = "epoch",
            save_strategy="epoch",  # no, epoch, steps
            overwrite_output_dir=True,  # use only one dir
            prediction_loss_only = True,
            # logging_dir='./logs',  
            # save_steps=5000,
        )
        trainer = Trainer(
            model = model,
            args = training_args,
            train_dataset = trainable_ds['train'],
            eval_dataset = trainable_ds['test'],
            # data_collator = data_collator,
            # config = model.config,
            # tokenizer = mydata.tokenizer
        )
        trainer.train()
        if bool(self.opt.save_model):
            trainer.save_model(opt.save_path)

    # def compute_metrics(eval_preds):
    #     metric = evaluate.load("seqeval")
    #     logits, labels = eval_preds
    #     predictions = np.argmax(logits, axis=-1)
    #     # Remove ignored index (special tokens)
    #     true_predictions = [
    #         [str(p) for (p, l) in zip(prediction, label) if l != -100]
    #         for prediction, label in zip(predictions, labels)
    #     ]
    #     true_labels = [
    #         [str(l) for (p, l) in zip(prediction, label) if l != -100]
    #         for prediction, label in zip(predictions, labels)
    #     ]

    #     return metric.compute(predictions=true_predictions, references=true_labels)


    def acc_and_f1(self, p):
        def simple_accuracy(preds, labels):
            return (preds == labels).mean().item()

        preds, labels = p

        preds = np.argmax(preds, axis=-1)
        acc = simple_accuracy(preds, labels)
        f1 = f1_score(y_true=labels, y_pred=preds,pos_label='positive', average="micro").item()
        return {
            "accuracy": acc,
            "f1": f1,
        }

    def compute_metrics(self, p):
        # sequence classification 
        if self.opt.task_type == 'sequence-classifier':
            return self.acc_and_f1(p)
        
        # token classification
        metric = evaluate.load("seqeval")

        predictions, labels = p
        predictions = np.argmax(predictions, axis=-1)

        # if self.opt.dataset_name == 'funsd':
        #     label_list = ['B-ANSWER', 'B-HEADER', 'B-QUESTION', 'I-ANSWER', 'I-HEADER', 'I-QUESTION', 'O']
        # elif self.opt.dataset_name == 'cord':
        #     label_list = ['B-MENU.CNT', 'B-MENU.DISCOUNTPRICE', 'B-MENU.ETC', 'B-MENU.ITEMSUBTOTAL', 'B-MENU.NM', 'B-MENU.NUM', 'B-MENU.PRICE', 'B-MENU.SUB_CNT', 'B-MENU.SUB_ETC', 'B-MENU.SUB_NM', 'B-MENU.SUB_PRICE', 'B-MENU.SUB_UNITPRICE', 'B-MENU.UNITPRICE', 'B-MENU.VATYN', 'B-SUB_TOTAL.DISCOUNT_PRICE', 'B-SUB_TOTAL.ETC', 'B-SUB_TOTAL.OTHERSVC_PRICE', 'B-SUB_TOTAL.SERVICE_PRICE', 'B-SUB_TOTAL.SUBTOTAL_PRICE', 'B-SUB_TOTAL.TAX_PRICE', 'B-TOTAL.CASHPRICE', 'B-TOTAL.CHANGEPRICE', 'B-TOTAL.CREDITCARDPRICE', 'B-TOTAL.EMONEYPRICE', 'B-TOTAL.MENUQTY_CNT', 'B-TOTAL.MENUTYPE_CNT', 'B-TOTAL.TOTAL_ETC', 'B-TOTAL.TOTAL_PRICE', 'B-VOID_MENU.NM', 'B-VOID_MENU.PRICE', 'I-MENU.CNT', 'I-MENU.DISCOUNTPRICE', 'I-MENU.ETC', 'I-MENU.NM', 'I-MENU.PRICE', 'I-MENU.SUB_ETC', 'I-MENU.SUB_NM', 'I-MENU.UNITPRICE', 'I-MENU.VATYN', 'I-SUB_TOTAL.DISCOUNT_PRICE', 'I-SUB_TOTAL.ETC', 'I-SUB_TOTAL.OTHERSVC_PRICE', 'I-SUB_TOTAL.SERVICE_PRICE', 'I-SUB_TOTAL.SUBTOTAL_PRICE', 'I-SUB_TOTAL.TAX_PRICE', 'I-TOTAL.CASHPRICE', 'I-TOTAL.CHANGEPRICE', 'I-TOTAL.CREDITCARDPRICE', 'I-TOTAL.EMONEYPRICE', 'I-TOTAL.MENUQTY_CNT', 'I-TOTAL.MENUTYPE_CNT', 'I-TOTAL.TOTAL_ETC', 'I-TOTAL.TOTAL_PRICE', 'I-VOID_MENU.NM']
        # elif self.opt.dataset_name == 'sorie':
        label_list = self.opt.label_list

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        # print('true predicts: ',true_predictions)
        # print('true labels: ',true_labels)
        results = metric.compute(predictions=true_predictions, references=true_labels)

        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

