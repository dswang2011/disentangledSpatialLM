
# from LMs.Roberta import RobertaClassifier
from LMs.LayoutLM import LayoutLMTokenclassifier
from LMs.LayoutLM import LayoutLM4DocVQA
from LMs.Roberta import GraphRobertaTokenClassifier, RobertaTokenClassifier
from LMs.SpatialLM import SpatialLMForMaskedLM, SpatialLMForTokenclassifier, SpatialLMConfig, SpatialLMForSequenceClassification, SpatialLMForDocVQA
from transformers import AutoConfig, AutoModel


def setup(opt):
    print('network:' + opt.network_type)
    # if opt.network_type == 'roberta':
    #     model = RobertaClassifier(opt)
    if opt.network_type == 'layoutlm':
        if opt.task_type == 'token-classifier':
            model = LayoutLMTokenclassifier(opt)
        elif opt.task_type == 'docvqa':
            model = LayoutLM4DocVQA(opt)
    elif opt.network_type == 'graph_roberta':
        model = GraphRobertaTokenClassifier(opt)
    elif opt.network_type == 'roberta':
        model = RobertaTokenClassifier(opt)
    elif opt.network_type == 'spatial_lm':
        if opt.task_type == 'mlm':
            # from_pretrained is put inside or outside
            if 'checkpoint_path' in opt.__dict__.keys():
                print('== load from the checkpoint === ', opt.checkpoint_path)
                config = SpatialLMConfig.from_pretrained(opt.checkpoint_path)   # borrow config
                model = SpatialLMForMaskedLM.from_pretrained(opt.checkpoint_path, config = config)
            else:
                # the first time, we first start from layoutlm; put layoutlm_dir
                print('=== load the first time from layoutlmv3 ===')
                config = AutoConfig.from_pretrained(opt.layoutlm_dir)   # borrow config
                model = SpatialLMForMaskedLM(config=config, start_dir_path=opt.layoutlm_dir)

        elif opt.task_type == 'token-classifier':
            config = SpatialLMConfig.from_pretrained(opt.checkpoint_path)
            config.num_labels=opt.num_labels    # set label num
            config.spatial_attention = opt.spatial_attention
            model = SpatialLMForTokenclassifier.from_pretrained(opt.checkpoint_path, config = config)
        elif opt.task_type == 'sequence-classifier':
            config = SpatialLMConfig.from_pretrained(opt.checkpoint_path)
            config.spatial_attention = opt.spatial_attention
            if not bool(opt.inference_only): # if it is train mode
                config.num_labels, config.id2label, config.label2id = opt.num_labels, opt.id2label, opt.label2id  # set label num
            model = SpatialLMForSequenceClassification.from_pretrained(opt.checkpoint_path, config=config)
        elif opt.task_type == 'docvqa':
            config = SpatialLMConfig.from_pretrained(opt.checkpoint_path)
            config.spatial_attention = opt.spatial_attention
            model = SpatialLMForDocVQA.from_pretrained(opt.checkpoint_path, config = config)
    elif opt.network_type == 'disentlm':
        if opt.task_type == 'mlm':
            # from_pretrained is put inside or outside
            if 'checkpoint_path' in opt.__dict__.keys():
                print('== load from the checkpoint === ', opt.checkpoint_path)
                config = SpatialLMConfig.from_pretrained(opt.checkpoint_path)   # borrow config
                model = SpatialLMForMaskedLM.from_pretrained(opt.checkpoint_path, config = config)
            else:
                # the first time, we first start from layoutlm; put layoutlm_dir
                print('=== load the first time from layoutlmv3 ===')
                config = AutoConfig.from_pretrained(opt.layoutlm_dir)   # borrow config
                model = SpatialLMForMaskedLM(config=config, start_dir_path=opt.layoutlm_dir)
    else:
        raise Exception('model not supported:{}'.format(opt.network_type))

    return model


