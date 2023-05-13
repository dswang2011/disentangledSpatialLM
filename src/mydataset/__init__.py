
def setup(opt):
    if opt.dataset_name.lower() == 'rvlcdip':
        from mydataset.rvlcdip import RVLCDIP as MyData
    elif opt.dataset_name.lower() == 'funsd_cord_sorie':
        from mydataset.funsd_cord_sorie import FUNSD_CORD_SORIE as MyData
    elif opt.dataset_name.lower() == 'funsd':
        from mydataset.funsd4lm import FUNSD as MyData
    elif opt.dataset_name.lower() == 'cord':
        from mydataset.cord4lm import CORD as MyData
    elif opt.dataset_name.lower() == 'rvl':
        from mydataset.rvl import RVL as MyData
    elif opt.dataset_name.lower() == 'sorie':
        from mydataset.sorie import SORIE as MyData
    elif opt.dataset_name.lower() == 'cdip':
        from mydataset.cdip import CDIP as MyData 
    elif opt.dataset_name.lower() == 'docvqa_ocr':
        from mydataset.docvqa_ocr import DocVQA as MyData

    else:
        raise Exception('dataset not supported:{}'.format(opt.dataset_name))

    mydataset = MyData(opt=opt)
    return mydataset
