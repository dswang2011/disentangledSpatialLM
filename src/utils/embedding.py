import spacy
# from spacy.tokenizer import Tokenizer
from transformers import LayoutLMTokenizer

from transformers import BertTokenizer, BertModel


class Embedding(object):
    def __init__(self, opt):
        self.opt = opt
        self.nlp = spacy.load('en_core_web_sm')
        # if opt.network_type == 'roberta':
        #     self.tokenizer = RobertaTokenizer.from_pretrained(opt.roberta_dir)
        # elif opt.network_type == 'gpt2':
        #     self.tokenizer = GPT2Tokenizer.from_pretrained(opt.gpt2_dir)
        # elif opt.network_type in ['word2vec']:
        #     self.nlp = spacy.load('en_core_web_lg')
        # elif opt.network_type == 'layoutlm':
        #     self.tokenizer = LayoutLMTokenizer.from_pretrained(opt.layoutlm_dir)
        # else:
        #     self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        #     self.model = BertModel.from_pretrained("bert-base-uncased")
            # self.vocab = self.tokenizer.vocab
            # print("VOCAB", len(self.vocab))
            # self.embedding_matrix,self.embedding_dict = self.get_embedding()


    def get_embedding_dict(self,GLOVE_DIR):
        embeddings_index = {}
        f = codecs.open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'),encoding="utf-8")
        for line in f:
            if line.strip()=='':
                continue
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        return embeddings_index

    def get_text_mean_vect(self,text):
        doc = self.nlp(text)
		# word vector is doc[idx].vector
        return doc.vector   # mean vector of the entire sentence

    def get_text_vect(self,text):
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs,output_hidden_states=False)
        vector = outputs.pooler_output.view(-1).detach().numpy()
        return vector

    def texts_to_seqences(self,sentences):
        """
			return e.g.,
			{'input_ids': tensor([[ 101, 8667,  146,  112,  182,  170, 1423, 5650,  102],
									[ 101, 1262, 1330, 5650,  102,    0,    0,    0,    0],
			'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0],
										[0, 0, 0, 0, 0, 0, 0, 0, 0],
			'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1],
										[1, 1, 1, 1, 1, 0, 0, 0, 0]])}
		"""
		# pt: pytorch tensor, tf: tensorflow tensor
        if self.opt.network_type == 'gpt2':
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        batch = self.tokenizer(sentences, padding=True, truncation=True, max_length=self.opt.max_seq_len, return_tensors="pt")
        return batch
        
if __name__=='__main__':
    emb = Embedding(opt=None)
    vect = emb.get_text_mean_vect('my name is dongsheng')
    print(vect)
    
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# model = BertModel.from_pretrained("bert-base-uncased")

# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# outputs = model(**inputs,output_hidden_states=True)

# last_hidden_states = outputs.last_hidden_state
# hidden_states = outputs.hidden_states
# print(outputs.keys())

# print(last_hidden_states)
# # print(hidden_states)

# print(outputs.pooler_output)