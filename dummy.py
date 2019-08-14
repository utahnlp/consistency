import sys
sys.path.insert(0, '../pytorch-pretrained-BERT')
import torch
from torch import cuda
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

torch.cuda.set_device(0)

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenized input
text1 = "[CLS] Which NFL team represented the AFC at Super Bowl 50 ? [SEP] Super Bowl 50 was an American football game to determine the champion of the National Football League ( NFL ) for the 2015 season . The American Football Conference ( AFC ) champion Denver Broncos defeated the National Football Conference ( NFC ) champion Carolina Panthers 24 – 10 to earn their third Super Bowl title . The game was played on February 7 , 2016 , at Levi ' s Stadium in the San Francisco Bay Area at Santa Clara , California . As this was the 50th Super Bowl , the league emphasized the \" golden anniversary \" with various gold - themed initiatives , as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals ( under which the game would have been known as \" Super Bowl L \" ) , so that the logo could prominently feature the Arabic numerals 50 . [SEP]"
text2 = "[CLS] In what year was the Joan B. Kroc Institute for International Peace Studies founded ? [SEP] Ctenophora ( / tᵻˈnɒfərə / ; singular ctenophore , / ˈtɛnəfɔːr / or / ˈtiːnəfɔːr / ; from the Greek κτείς kteis ' comb ' and φέρω pherō ' carry ' ; commonly known as comb jellies ) is a phylum of animals that live in marine waters worldwide . Their most distinctive feature is the ‘ combs ’ – groups of cilia which they use for swimming – they are the largest animals that swim by means of cilia . Adults of various species range from a few millimeters to 1.5 m ( 4 ft 11 in ) in size . Like cnidarians , their bodies consist of a mass of jelly , with one layer of cells on the outside and another lining the internal cavity . In ctenophores , these layers are two cells deep , while those in cnidarians are only one cell deep . Some authors combined ctenophores and cnidarians in one phylum , Coelenterata , as both groups rely on water flow through the body cavity for both digestion and respiration . Increasing awareness of the differences persuaded more recent authors to classify them as separate phyla . [SEP]"

seqs = [text1, text2]


seg_idx = []
token_idx = []
max_seq_l = 0
for seq in seqs:
	tokenized_text = tokenizer.tokenize(seq)
	print(tokenized_text)
	mid_boundary = tokenized_text.index('[SEP]')
	seg_idx.append([0 for _ in range(mid_boundary+1)] + [1 for _ in range(len(tokenized_text)-mid_boundary-1)])
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
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()
print('evaluation done.')


token_tensor = token_tensor.cuda()
seg_tensor = seg_tensor.cuda()
att_tensor = att_tensor.cuda()

print(token_tensor.type(), seg_tensor.type(), att_tensor.type())

model = model.cuda()

batch_cnt = 100
print('parsing for {0} batches...'.format(batch_cnt))
for _ in range(batch_cnt):
	# Predict hidden states features for each layer
	with torch.no_grad():
	    encoded_layers, pool_layers = model(token_tensor, seg_tensor, att_tensor, output_all_encoded_layers=False)
	
	print(encoded_layers.shape)
	print(pool_layers.shape)
	#for l in encoded_layers:
	#	print(l.shape)


