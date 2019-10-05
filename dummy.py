import sys
sys.path.insert(0, '../transformers')
import torch
from torch import cuda
from transformers import *

def get_cls(key):
	model_map={"bert-base-uncased": (BertModel, BertTokenizer),
		"gpt2": (GPT2Model, GPT2Tokenizer),
		"roberta-base": (RobertaModel, RobertaTokenizer)}
	return model_map[key]

torch.cuda.set_device(0)
bert_type = "roberta-base"
model_cls, tokenizer_cls = get_cls(bert_type)

# Load pre-trained model tokenizer (vocabulary)
tokenizer = tokenizer_cls.from_pretrained(bert_type)
CLS = tokenizer.cls_token
SEP = tokenizer.sep_token
SEG1, SEG2 = (0, 0) if 'roberta' in bert_type else (0, 1)

# Tokenized input
text1 = "{0} Which NFL team represented the AFC at Super Bowl 50 ? {1} Super Bowl 50 was an American football game to determine the champion of the National Football League ( NFL ) for the 2015 season . The American Football Conference ( AFC ) champion Denver Broncos defeated the National Football Conference ( NFC ) champion Carolina Panthers 24 – 10 to earn their third Super Bowl title . The game was played on February 7 , 2016 , at Levi ' s Stadium in the San Francisco Bay Area at Santa Clara , California . As this was the 50th Super Bowl , the league emphasized the \" golden anniversary \" with various gold - themed initiatives , as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals ( under which the game would have been known as \" Super Bowl L \" ) , so that the logo could prominently feature the Arabic numerals 50 . {1}".format(CLS, SEP)
text2 = "{0} In what year was the Joan B. Kroc Institute for International Peace Studies founded ? {1} Ctenophora ( / tᵻˈnɒfərə / ; singular ctenophore , / ˈtɛnəfɔːr / or / ˈtiːnəfɔːr / ; from the Greek κτείς kteis ' comb ' and φέρω pherō ' carry ' ; commonly known as comb jellies ) is a phylum of animals that live in marine waters worldwide . Their most distinctive feature is the ‘ combs ’ – groups of cilia which they use for swimming – they are the largest animals that swim by means of cilia . Adults of various species range from a few millimeters to 1.5 m ( 4 ft 11 in ) in size . Like cnidarians , their bodies consist of a mass of jelly , with one layer of cells on the outside and another lining the internal cavity . In ctenophores , these layers are two cells deep , while those in cnidarians are only one cell deep . Some authors combined ctenophores and cnidarians in one phylum , Coelenterata , as both groups rely on water flow through the body cavity for both digestion and respiration . Increasing awareness of the differences persuaded more recent authors to classify them as separate phyla . {1}".format(CLS, SEP)

seqs = [text1, text2]


seg_idx = []
token_idx = []
max_seq_l = 0
for seq in seqs:
	tokenized_text = tokenizer.tokenize(seq)
	print(tokenized_text)
	mid_boundary = tokenized_text.index(SEP)
	seg_idx.append([SEG1 for _ in range(mid_boundary+1)] + [SEG2 for _ in range(len(tokenized_text)-mid_boundary-1)])
	token_idx.append(tokenizer.convert_tokens_to_ids(tokenized_text))

	assert(len(seg_idx[-1]) == len(token_idx[-1]))
	max_seq_l = max(max_seq_l, len(token_idx[-1]))

att_mask = []
for s, t in zip(seg_idx, token_idx):
	att_mask.append([1 for _ in range(len(s))])
	if len(s) < max_seq_l:
		att_mask[-1].extend([0 for _ in range(max_seq_l-len(s))])
		s.extend([0 for _ in range(max_seq_l-len(s))])
		t.extend([0 for _ in range(max_seq_l-len(t))])
	assert(len(s) == len(att_mask[-1]))

# Convert inputs to PyTorch tensors
token_tensor = torch.tensor(token_idx)
seg_tensor = torch.tensor(seg_idx)
att_tensor = torch.tensor(att_mask)

print('loading BERT model...')
model = model_cls.from_pretrained(bert_type)
model.eval()
print('evaluation done.')

gpuid=-1

if gpuid != -1:
	token_tensor = token_tensor.cuda(1)
	seg_tensor = seg_tensor.cuda(1)
	att_tensor = att_tensor.cuda(1)
	model = model.cuda(1)

print(token_tensor.type(), seg_tensor.type(), att_tensor.type())
print(token_tensor.shape, seg_tensor.shape, att_tensor.shape)
print(model.embeddings.position_embeddings.weight.shape)
print(token_tensor.max(), token_tensor.min())
print(att_tensor)

batch_cnt = 2
print('parsing for {0} batches...'.format(batch_cnt))
for _ in range(batch_cnt):
	# Predict hidden states features for each layer
	with torch.no_grad():
	    encoded_layers, pool_layers = model(token_tensor, seg_tensor, None)
	
	print(encoded_layers.shape)
	print(pool_layers.shape)
	#for l in encoded_layers:
	#	print(l.shape)


