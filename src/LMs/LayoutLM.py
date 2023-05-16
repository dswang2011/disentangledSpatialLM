# -*- coding: utf-8 -*-

import torch.nn as nn

# from transformers import RobertaModel, RobertaConfig
from transformers import LayoutLMForTokenClassification, AutoModelForTokenClassification
from LMs.layoutlmv3 import LayoutLMv3ForTokenClassification
from transformers import AutoModelForQuestionAnswering, AutoConfig


class LayoutLM4DocVQA(nn.Module):
    def __init__(self,opt,freeze_bert=False):
        super(LayoutLM4DocVQA, self).__init__()
        self.opt = opt

        self.layoutlm = AutoModelForQuestionAnswering.from_pretrained(opt.layoutlm_dir)

    # dif: token_type_ids; start and end positions -> labels; image -> no pixel_values
    def forward(self,input_ids, token_type_ids, bbox, attention_mask, pixel_values, start_positions=None, end_positions=None,**args):
        # print('input_ids:',input_ids.shape)
        # print('token_type_ids:',token_type_ids.shape)
        # print('bbox:',bbox.shape)
        # print('attention_mask:',attention_mask.shape)
        # print('pixel_values:',pixel_values.shape)

        outputs = self.layoutlm(
            input_ids = input_ids,
            token_type_ids = token_type_ids,
            bbox = bbox,
            attention_mask = attention_mask,
            pixel_values = pixel_values, 
            start_positions = start_positions,
            end_positions = end_positions
        )
        return outputs


class LayoutLMTokenclassifier(nn.Module):
    def __init__(self, opt, freeze_bert=False):
        super(LayoutLMTokenclassifier, self).__init__()
        self.opt = opt
        # self.config = RobertaConfig.from_pretrained(opt.roberta_dir)
        # self.roberta = RobertaModel(self.config)
        self.config = AutoConfig.from_pretrained(opt.layoutlm_dir)
        self.config.visual_embed = False
        self.config.num_labels = opt.num_labels
        print('== vis == ', self.config.visual_embed)
        # self.layoutlm = AutoModelForTokenClassification.from_pretrained(opt.layoutlm_dir, config = self.config, num_labels=opt.num_labels, label2id=opt.label2id, id2label=opt.id2label)
        self.layoutlm = LayoutLMv3ForTokenClassification.from_pretrained(opt.layoutlm_dir, config = self.config)

        # freeze the bert model
        if freeze_bert:
            for param in self.roberta.parameters():
                param.requires_grad = False
        

    def forward(self,input_ids, bbox, attention_mask, pixel_values, labels, **args):
        outputs = self.layoutlm(input_ids = input_ids, bbox = bbox, attention_mask = attention_mask, pixel_values = pixel_values, labels = labels)
        # outputs = self.layoutlm(input_ids = input_ids, bbox = None, attention_mask = attention_mask, pixel_values = None, labels = labels)
        return outputs

