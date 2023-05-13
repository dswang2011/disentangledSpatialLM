import os
import json
from OCRs import tesseract4img
from datasets import load_from_disk

def get_imgs_dfs(dir, suffix = 'png'):
    res = []
    files = os.listdir(dir)
    for file in files:
        file_path = os.path.join(dir,file)

        if os.path.isdir(file_path):
            sub_res = get_imgs_dfs(file_path, suffix)
            res += sub_res
        elif file.endswith(suffix):
            res.append(file_path)
    return res


def get_question_pairs(base,split='val'):
    # from json of questions and answers
    file_path = os.path.join(base, split, split+'_v1.0.json')
    
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

        ucsf_doc_id = sample['ucsf_document_id']   # e.g.,: txpp0227
        ucsf_doc_page = sample['ucsf_document_page_no'] # e.g.,: 10
        docID_page = ucsf_doc_id + '_' + ucsf_doc_page
        trip_object = (docID_page, question, answers)
        id2trip[qID] = trip_object
    return id2trip

def generate_docvqa_ds():
    dir = '/home/ubuntu/air/vrdu/datasets/docvqa'

    for split in ['train','val','test']:
        files,labels = get_img_label_pairs(split)
        print(split, ' to be generated file num:',len(files))

        img_dir = os.path.join(dir,split, 'documents')
        img_paths = get_img_dfs(img_dir)

        mydataset = tesseract4img.imgs_to_dataset_generator(sub_files,sub_labels)
        saveto = '/home/ubuntu/air/vrdu/datasets/rvl_HF_datasets/full_rvl_'+split+str(i)+'_dataset.hf'
        mydataset.save_to_disk(saveto)
        print(mydataset)


# find the index, of the answer for raw tokens;
def _subfinder(words_list, answer_list):  
    # print('input words:',words_list)
    # print('input ans:',answer_list)
    matches = []
    start_indices = []
    end_indices = []
    for idx, i in enumerate(range(len(words_list))):
        if words_list[i] == answer_list[0] and words_list[i:i+len(answer_list)] == answer_list:
            matches.append(answer_list)
            start_indices.append(idx)
            end_indices.append(idx + len(answer_list) - 1)
    if matches:
        return matches[0], start_indices[0], end_indices[0]
    else:
        return None, 0, 0
def _raw_ans_word_idx_range(tokens, answers):
    # Match trial 1: try to find one of the answers in the context, return first match
    lower_tokens = [token.lower() for token in tokens]
    answers = sorted(answers, key=len, reverse=True)    # longest to shortest
    for answer in answers:
        match, ans_word_idx_start, ans_word_idx_end = _subfinder(lower_tokens, answer.lower().split())
        if match:
            break
    return match, ans_word_idx_start, ans_word_idx_end 

def get_start_end_ds(ds):
    def start_end_map(sample):
        tokens = sample['tokens']
        answers = sample['answers']
        match, ans_word_idx_start, ans_word_idx_end = _raw_ans_word_idx_range(tokens, answers)
        sample['ans_token_start'] = ans_word_idx_start
        sample['ans_token_end'] = ans_word_idx_end
        return sample
    ans_ds = ds.map(start_end_map,num_proc=os.cpu_count())  # remove_columns=['answers']
    return ans_ds

def ans_exists(sample):
    return sample['ans_token_start']==0 and sample['ans_token_end']==0

if __name__=='__main__':
    split = 'train'   # train, test, val

    if False:
        # load qa pairs     
        base = '/home/ubuntu/air/vrdu/datasets/docvqa'
        # 1 load all QA pairs 
        id2trip = get_question_pairs(base,split)    

        cnt = 0

        img_paths = []
        ans_list = []
        q_list = []

        for k,val in id2trip.items():
            docID_page, question, answers = val
            img_path = os.path.join(base, split, 'documents', docID_page +'.png')       
            ans_list.append(answers)
            q_list.append(question)
            img_paths.append(img_path)
            if not question: continue

        # 2 parse imgs and generate ds
        ds = tesseract4img.imgs_to_dataset_generator(img_paths,labels=None, tesseract_wait=True, questions = q_list, answers = ans_list)
        print(ds)

        # 3 output
        ds.save_to_disk(split+'.hf') # 5304 samples
    else:
        ds = load_from_disk('/home/ubuntu/air/vrdu/datasets/docvqa/hfs/' + split+'.hf')
        # 4. map to find the answers (from longest to shortest)
        ans_ds = get_start_end_ds(ds)   # 
        ans_ds = ans_ds.filter(ans_exists)  # filter empty answers
        print(ans_ds)
        ans_ds.save_to_disk('/home/ubuntu/air/vrdu/datasets/docvqa/hfs/' + split+'_ans.hf')
