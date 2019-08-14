Implementation of the NLI model in our EMNLP 2019 paper: [A Logic-Driven Framework for Consistency of Neural Models]
```
@inproceedings{li2019consistency,
      author    = {Li, Tao and Gupta, Vivek and Mehta, Maitrey and Srikumar, Vivek},
      title     = {A Logic-Driven Framework for Consistency of Neural Models},
      booktitle = {2019 Conference on Empirical Methods in Natural Language Processing },
      year      = {2019}
  }
```

### 0. Prerequisites
Have the following installed:
```
python 3.6+
pytorch 1.0
h5py
numpy
pytorch BERT by huggingface(https://github.com/huggingface/pytorch-pretrained-BERT)
	(download and put in ../pytorch-pretrained-BERT, not necessarily installed)
	(However, for exact reproducibility, use the pytorch-pretrained-BERT.zip in this repo)
glove.840B.300d.txt (under ./data/)
	(We don't actually use it, but need it for preprocessing.)
```

Besides above, make sure snli_1.0 data is unpacked to ```./data/bert_nli/```, e.g. ```./data/bert_nli/snli_1.0_dev.txt```.

And have mnli data unpacked to ```./data/bert_nli/```. We will use the ```mnli_dev_matched``` for validation, and the ```mnli_dev_mismatched``` for testing.

Unpack the ```./data/bert_nli/mscoco.zip``` and move all the files to the ```./data/bert_nli/``` directory, e.g. ```./data/bert_nli/mscoco.raw.sent1.txt```.


### 1. Preprocessing
Preprocessing of snli is separated into the following steps.
```
python3 preprocess.py --glove ./data/glove.840B.300d.txt --batch_size 48 --dir ./data/bert_nli/ --output snli
python3 get_char_idx.py --dict snli.allword.dict --token_l 16 --freq 5 --output char
```

**NOTE** For exact reproducibility, we will use the ```dev_excl_anno``` for actual snli validation. The difference between this and the official development set is that we reserved ```1000``` examples for manual analysis. These examples are later excluded from experiments to avoid contamination.

Preprocessing of mnli dataset:
```
python3 preprocess.py --glove ./data/glove.840B.300d.txt --batch_size 36 --dir ./data/bert_nli/ \
	--sent1 mnli_train.sent1.txt --sent2 mnli_train.sent2.txt --label mnli_train.label.txt \
	--sent1_val mnli_dev_matched.sent1.txt --sent2_val mnli_dev_matched.sent2.txt --label_val mnli_dev_matched.label.txt \
	--sent1_test mnli_dev_mismatched.sent1.txt --sent2_test mnli_dev_mismatched.sent2.txt --label_test mnli_dev_mismatched.label.txt \
	--tokenizer_output mnli --output mnli --max_seq_l 500
```


Preprocessing of mscoco dataset:
```
python3 extra_preprocess.py --glove ./data/glove.840B.300d.txt --batch_size 48 --dir ./data/bert_nli/ --sent1 mscoco.raw.sent1.txt --sent2 mscoco.raw.sent2.txt --sent3 mscoco.raw.sent3.txt --tokenizer_output mscoco --output mscoco
python3 extra_preprocess.py --glove ./data/glove.840B.300d.txt --batch_size 48 --dir ./data/bert_nli/ --sent1 mscoco.test.raw.sent1.txt --sent2 mscoco.test.raw.sent2.txt --sent3 mscoco.test.raw.sent3.txt --tokenizer_output mscoco.test --output mscoco.test
```

### 2. BERT Baseline

**Finetuning once** on both snli and mnli
```
mkdir models

GPUID=[GPUID]
LR=0.00003
PERC=1
for SEED in `seq 1 3`; do
	CUDA_VISIBLE_DEVICES=$GPUID python3 -u train.py --gpuid 0 --bert_gpuid 0 --dir ./data/bert_nli/ \
	--train_data mnli.train.hdf5 --val_data mnli.val.hdf5 --extra_train_data snli.train.hdf5 --extra_val_data snli.val.hdf5 \
	--learning_rate $LR --epochs 3 --warmup_epoch 3 \
	--enc bert --cls linear --hidden_size 768 --percent $PERC --dropout 0.1 \
	--fix_bert 0 --optim adam_fp16 --fp16 1 --seed ${SEED} \
	--save_file models/scratch_mnli_snli_perc${PERC//.}_seed${SEED} | tee models/scratch_mnli_snli_perc${PERC//.}_seed${SEED}.txt
done
```
Change ```[GPUID]``` to the desired device id, ```PERC``` specifies percentages of training data to use (1 is 100%). The above script will initiate BERT baselines with three different random seeds (i.e. three runs in a row). Expect to see exactly the same accuracy as we reported in our paper.

**Finetuning twice** on both snli and mnli

```
GPUID=[GPUID]
LR=0.00001
PERC=1
for SEED in `seq 1 3`; do
CUDA_VISIBLE_DEVICES=$GPUID python3 -u train.py --gpuid 0 --bert_gpuid 0 --dir ./data/bert_nli/ \
	--train_data mnli.train.hdf5 --val_data mnli.val.hdf5 --extra_train_data snli.train.hdf5 --extra_val_data snli.val.hdf5 \
	--learning_rate $LR --epochs 3 --warmup_epoch 3 \
	--enc bert --cls linear --hidden_size 768 --percent $PERC --dropout 0.1 \
	--fix_bert 0 --optim adam_fp16 --fp16 1 --seed ${SEED} \
	--load_file models/scratch_mnli_snli_perc${PERC//.}_seed${SEED} \
	--save_file models/twice_scratch_mnli_snli_perc${PERC//.}_seed${SEED} | tee models/twice_scratch_mnli_snli_perc${PERC//.}_seed${SEED}.txt
done
```
This will load the previously finetuned model and continue finetune with lowered learning rate. Expect to see exactly the same accuracy as we reported in our paper.

**Evaluating** on mirror consistency
```
GPUID=[GPUID]
PERC=1
for SWAP_SENT in 0 1; do
for SEED in `seq 1 3`; do
CUDA_VISIBLE_DEVICES=$GPUID python3 -u eval.py --gpuid 0 --bert_gpuid 0 --dir data/bert_nli/ --data mscoco.test.hdf5 \
	--enc bert --cls linear --hidden_size 768 --fp16 1 --dropout 0.0 --swap_sent $SWAP_SENT \
	--pred_output models/twice_scratch_mnli_snli_perc${PERC//.}_seed${SEED}_swap${SWAP_SENT} \
	--load_file models/twice_scratch_mnli_snli_perc${PERC//.}_seed${SEED} | tee models/twice_scratch_mnli_snli_perc${PERC//.}_seed${SEED}.evallog.txt
done
done
```

**Evaluating** on transitivity consistency
```
GPUID=[GPUID]
PERC=1
for PAIR in alpha beta gamma; do
for SEED in `seq 1 3`; do
CUDA_VISIBLE_DEVICES=$GPUID python3 -u eval.py --gpuid 0 --bert_gpuid 0 --dir data/bert_nli/ --data mscoco.test.hdf5 \
	--enc bert --cls linear --hidden_size 768 --fp16 1 --dropout 0.0 --data_triple_mode 1 --sent_pair $PAIR --swap_sent 0 \
	--pred_output models/twice_scratch_mnli_snli_perc${PERC//.}_seed${SEED}_${PAIR} \
	--load_file models/twice_scratch_mnli_snli_perc${PERC//.}_seed${SEED} | tee models/twice_scratch_mnli_snli_perc${PERC//.}_seed${SEED}.evallog.txt
done
done
```

### 3. BERT+M

```
GPUID=[GPUID]
LR=0.00001
CONSTR=6
PERC=1
LAMBD=1
for SEED in `seq 1 3`; do
	CUDA_VISIBLE_DEVICES=$GPUID python3 -u train.py --gpuid 0 --bert_gpuid 0 --dir ./data/bert_nli/ \
	--train_data mnli.train.hdf5 --val_data mnli.val.hdf5 --extra_train_data snli.train.hdf5 --extra_val_data snli.val.hdf5 \
	--learning_rate $LR --epochs 3 --warmup_epoch 3 \
	--loss transition --fwd_mode flip --lambd ${LAMBD} \
	--enc bert --cls linear --hidden_size 768 --percent $PERC --dropout 0.1 --constr ${CONSTR} \
	--fix_bert 0 --optim adam_fp16 --fp16 1 --seed ${SEED} \
	--load_file models/scratch_mnli_snli_perc${PERC//.}_seed${SEED} \
	--save_file models/both_flip${CONSTR//,}_lr${LR//.}_lambd${LAMBD//.}_perc${PERC//.}_seed${SEED} | tee models/both_flip${CONSTR//,}_lr${LR//.}_lambd${LAMBD//.}_perc${PERC//.}_seed${SEED}.txt
done
```
Do change ```PERC``` and ```LAMBD``` accordingly.

**Evaluating** on mirror consistency
```
GPUID=[GPUID]
LR=0.00001
CONSTR=6
PERC=0.2
LAMBD=1
for SWAP_SENT in 0 1; do
for SEED in `seq 1 3`; do
	CUDA_VISIBLE_DEVICES=$GPUID python3 -u eval.py --gpuid 0 --bert_gpuid 0 --dir ./data/bert_nli/ --data mscoco.test.hdf5 \
	--enc bert --cls linear --dropout 0.0 --hidden_size 768 --fp16 1 --data_triple_mode 0 --swap_sent $SWAP_SENT \
	--pred_output models/both_flip${CONSTR//,}_lr${LR//.}_lambd${LAMBD//.}_perc${PERC//.}_seed${SEED}_swap${SWAP_SENT} \
	--load_file models/both_flip${CONSTR//,}_lr${LR//.}_lambd${LAMBD//.}_perc${PERC//.}_seed${SEED} | tee models/both_flip${CONSTR//,}_lr${LR//.}_lambd${LAMBD//.}_perc${PERC//.}_seed${SEED}.triplelog.txt
done
done

python3 confusion_table.py --log both_flip${CONSTR//,}_lr${LR//.}_lambd${LAMBD//.}_perc${PERC//.}
```

**Evaluating** on transitivity consistency
```
GPUID=[GPUID]
LR=0.00001
CONSTR=6
PERC=0.2
LAMBD=1
for PAIR in alpha beta gamma; do
for SEED in `seq 1 3`; do
	CUDA_VISIBLE_DEVICES=$GPUID python3 -u eval.py --gpuid 0 --bert_gpuid 0 --dir ./data/bert_nli/ --data mscoco.test.hdf5 \
	--enc bert --cls linear --dropout 0.0 --hidden_size 768 --fp16 1 --data_triple_mode 1 --sent_pair $PAIR \
	--pred_output models/both_flip${CONSTR//,}_lr${LR//.}_lambd${LAMBD//.}_perc${PERC//.}_seed${SEED}_${PAIR} \
	--load_file models/both_flip${CONSTR//,}_lr${LR//.}_lambd${LAMBD//.}_perc${PERC//.}_seed${SEED} | tee models/both_flip${CONSTR//,}_lr${LR//.}_lambd${LAMBD//.}_perc${PERC//.}_seed${SEED}.triplelog.txt
done
done

for SEED in `seq 1 3`; do
	python3 triple_confusion.py --log both_flip${CONSTR//,}_lr${LR//.}_lambd${LAMBD//.}_perc${PERC//.} --seed $SEED
done
```

### 4. BERT+M,U

```
GPUID=[GPUID]
PERC=0.01
PERC_U=0.25
CONSTR=6
LR=0.000005
LAMBD=1
LAMBD_P=0.001
for SEED in `seq 1 3`; do
CUDA_VISIBLE_DEVICES=$GPUID python3 -u train.py --gpuid 0 --bert_gpuid 0 --dir ./data/bert_nli/ \
	--train_data mnli.train.hdf5 --val_data mnli.val.hdf5 --extra_train_data snli.train.hdf5 --extra_val_data snli.val.hdf5 \
	--unlabeled_data mscoco.hdf5 --unlabeled_triple_mode 0 \
	--loss transition --fwd_mode flip_and_unlabeled --lambd ${LAMBD} \
	--learning_rate $LR --epochs 3 --warmup_epoch 3 --dropout 0.1 --constr ${CONSTR} \
	--enc bert --cls linear --hidden_size 768 --percent $PERC --unlabeled_perc ${PERC_U} --lambd_p $LAMBD_P \
	--fix_bert 0 --optim adam_fp16 --fp16 1 --seed ${SEED} \
	--load_file models/scratch_mnli_snli_perc${PERC//.}_seed${SEED} \
	--save_file models/both_mscoco_flip${CONSTR//,}_lr${LR//.}_lambd${LAMBD//.}_${LAMBD_P//.}_perc${PERC//.}_${PERC_U//.}_seed${SEED} | tee models/both_mscoco_flip${CONSTR//,}_lr${LR//.}_lambd${LAMBD//.}_${LAMBD_P//.}_perc${PERC//.}_${PERC_U//.}_seed${SEED}.txt
done
```
Do change ```PERC```, ```LAMBD```, and ```LAMBD_P``` accordingly.  For evaluation, construct evaluation script accordingly as above.


### 5. BERT+M,U,T

```
GPUID=[GPUID]
PERC=0.01
PERC_U=0.25
CONSTR=1,2,3,4,6
LR=0.000005
LAMBD=1
LAMBD_P=0.00001
LAMBD_T=0.000001
for SEED in `seq 3 3`; do
CUDA_VISIBLE_DEVICES=$GPUID python3 -u train.py --gpuid 0 --bert_gpuid 0 --dir ./data/bert_nli/ \
	--train_data mnli.train.hdf5 --val_data mnli.val.hdf5 --extra_train_data snli.train.hdf5 --extra_val_data snli.val.hdf5 \
	--unlabeled_data mscoco.hdf5 --unlabeled_triple_mode 1 \
	--loss transition --fwd_mode flip_and_triple --fix_bert 0 --optim adam_fp16 --fp16 1 --weight_decay 1 \
	--learning_rate $LR --epochs 3 --warmup_epoch 3 --dropout 0.1 --constr ${CONSTR} \
	--enc bert --cls linear --hidden_size 768 --percent $PERC --unlabeled_perc ${PERC_U} --lambd ${LAMBD} --lambd_p $LAMBD_P --lambd_t $LAMBD_T \
	--seed ${SEED} \
	--load_file models/scratch_mnli_snli_perc${PERC//.}_seed${SEED} \
	--save_file models/both_mscoco_flip_triple${CONSTR//,}_lr${LR//.}_lambd${LAMBD//.}_${LAMBD_P//.}_${LAMBD_T//.}_perc${PERC//.}_${PERC_U//.}_seed${SEED} | tee models/both_mscoco_flip_triple${CONSTR//,}_lr${LR//.}_lambd${LAMBD//.}_${LAMBD_P//.}_${LAMBD_T//.}_perc${PERC//.}_${PERC_U//.}_seed${SEED}.txt
done
```
Do change ```PERC```, ```LAMBD```, and ```LAMBD_P``` accordingly. For evaluation, construct evaluation script accordingly as above.

## Issues & To-dos
- [ ] Sanity check
