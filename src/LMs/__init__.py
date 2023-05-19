
# from LMs.Roberta import RobertaClassifier
from LMs.LayoutLM import LayoutLMTokenclassifier
from LMs.LayoutLM import LayoutLM4DocVQA
from LMs.Roberta import GraphRobertaTokenClassifier, RobertaTokenClassifier
# from LMs.SpatialLM import SpatialLMForMaskedLM, SpatialLMForTokenclassifier, SpatialLMConfig, SpatialLMForSequenceClassification, SpatialLMForDocVQA
from transformers import AutoConfig, AutoModel
from LMs.config_DisentLM import DisentLMConfig
from LMs.disentlm import DisentLMForTokenClassification,DisentLMForMaskedLM

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

    elif opt.network_type == 'disentlm':
        if opt.task_type == 'mlm':
            # from_pretrained is put inside or outside
            if 'checkpoint_path' in opt.__dict__.keys():
                print('== load from the checkpoint === ', opt.checkpoint_path)
                config = DisentLMConfig.from_pretrained(opt.checkpoint_path)   # borrow config
                config.entangle_mode = opt.entangle_mode
                config.visual_embed = False
                model = DisentLMForMaskedLM.from_pretrained(opt.checkpoint_path, config = config)
            else:
                # the first time, we first start from layoutlm; put layoutlm_dir
                print('=== load the first time from layoutlmv3 ===')
                config = AutoConfig.from_pretrained(opt.layoutlm_dir)   # borrow config
                model = SpatialLMForMaskedLM(config=config, start_dir_path=opt.layoutlm_dir)
        elif opt.task_type == 'token-classifier':
            config = DisentLMConfig.from_pretrained(opt.checkpoint_path)
            config.num_labels=opt.num_labels    # set label num
            # config.id2label, config.label2id = opt.id2label, opt.label2id
            config.entangle_mode = opt.entangle_mode
            config.visual_embed = False
            model = DisentLMForTokenClassification(config = config)
        print('entangle mode:', config.entangle_mode)

    else:
        raise Exception('model not supported:{}'.format(opt.network_type))

    return model


