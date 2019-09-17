<p align="center"><img width="60%" src="logo.png" /></p>
---
### What is special in this branch?
This branch accommodates recent updates in huggingface's pytorch transformer hub and nvidia apex.
The ```master``` branch is customized for bert-base while this branch works with other typical bertology models in the huggingface hub.
To use that, start from the prerequisites steps.

---

Implementation of the NLI model in our EMNLP 2019 paper: [A Logic-Driven Framework for Consistency of Neural Models](https://arxiv.org/abs/1909.00126)
```
@inproceedings{li2019consistency,
      author    = {Li, Tao and Gupta, Vivek and Mehta, Maitrey and Srikumar, Vivek},
      title     = {A Logic-Driven Framework for Consistency of Neural Models},
      booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing},
      year      = {2019}
  }
```

### 0. Prerequisites

**[Hardware]**
All of our BERT models are based on BERT base version. The batch size, sequence length, and data format are configurated to run smoothly on CUDA device with 8GB memory.

Have the following installed:
```
python 3.6+
NVCC compiler 10.0
pytorch 1.2.0
h5py
numpy
spacy 2.0.11 (with en model)
nvidia apex
pytorch BERT by huggingface(https://github.com/huggingface/pytorch-transformers.git)
	(download and put in ../pytorch-transformers, not necessarily installed)
glove.840B.300d.txt (under ./data/)
	(We don't actually use it, but need it for preprocessing (due to an old design).)
```
NOTE, the pytorch and apex need to use the same version of nvcc for correct installation.

**[SNLI]**
Besides above, make sure snli_1.0 data is unpacked to ```./data/bert_nli/```, e.g. ```./data/bert_nli/snli_1.0_train.txt```.


**[MNLI]**
And have mnli_1.0 data unpacked to ```./data/bert_nli/```. We will use the ```mnli_dev_matched``` for validation, and the ```mnli_dev_mismatched``` for testing. For example, the validation file should be at ```./data/bert_nli/multinli_1.0_dev_matched.txt```

**[MSCOCO]**
Unpack mscoco sample data via ```unzip ./data/bert_nli/mscoco.zip```. The zip file contains training split (e.g. ```mscoco.raw.sent1.txt```) with ```400k``` sentence triples and test split (e.g. ```mscoco.test.raw.sent1.txt```) with ```100k``` sentence triples. In practice, our paper sampled ```100k``` (i.e. ```25%```) from the training split, and used all examples in the test split.


### 1. Preprocessing

**[SNLI]**
Preprocessing of SNLI is separated into the following steps.
```
python3 snli_extract.py --data ./data/bert_nli/snli_1.0_train.txt --output ./data/bert_nli/train
python3 snli_extract.py --data ./data/bert_nli/snli_1.0_test.txt --output ./data/bert_nli/test

python3 preprocess.py --glove ./data/glove.840B.300d.txt --batch_size 40 --max_seq_l 450 --bert_type roberta-base --dir ./data/bert_nli/ --output snli --tokenizer_output snli
python3 get_char_idx.py --dict ./data/bert_nli/snli.allword.dict --token_l 16 --freq 5 --output char
```

NOTE, For exact reproducibility, we will use the ```dev_excl_anno.raw.sent*.txt``` for actual SNLI validation. These files are already included in the ```./data/bert_nli/``` directory and will be implicitly used in the above scripts. The difference is that we reserved ```1000``` examples for preliminary manual analysis and then later excluded them from experiments to avoid contamination.


**[MNLI]**
Preprocessing of MNLI dataset:
```
python3 mnli_extract.py --data ./data/bert_nli/multinli_1.0_dev_mismatched.txt --output ./data/bert_nli/mnli.test
python3 mnli_extract.py --data ./data/bert_nli/multinli_1.0_train.txt --output ./data/bert_nli/mnli.train
python3 mnli_extract.py --data ./data/bert_nli/multinli_1.0_dev_matched.txt --output ./data/bert_nli/mnli.dev

python3 preprocess.py --glove ./data/glove.840B.300d.txt --batch_size 30 --bert_type roberta-base --dir ./data/bert_nli/ \
	--sent1 mnli.train.raw.sent1.txt --sent2 mnli.train.raw.sent2.txt --label mnli.train.label.txt \
	--sent1_val mnli.dev.raw.sent1.txt --sent2_val mnli.dev.raw.sent2.txt --label_val mnli.dev.label.txt \
	--sent1_test mnli.test.raw.sent1.txt --sent2_test mnli.test.raw.sent2.txt --label_test mnli.test.label.txt \
	--tokenizer_output mnli --output mnli --max_seq_l 500
```

**[MSCOCO]**
Preprocessing of mscoco dataset:
```
python3 extra_preprocess.py --glove ./data/glove.840B.300d.txt --batch_size 40 --bert_type roberta-base --dir ./data/bert_nli/ --sent1 mscoco.raw.sent1.txt --sent2 mscoco.raw.sent2.txt --sent3 mscoco.raw.sent3.txt --tokenizer_output mscoco --output mscoco
python3 extra_preprocess.py --glove ./data/glove.840B.300d.txt --batch_size 48 --bert_type roberta-base --dir ./data/bert_nli/ --sent1 mscoco.test.raw.sent1.txt --sent2 mscoco.test.raw.sent2.txt --sent3 mscoco.test.raw.sent3.txt --tokenizer_output mscoco.test --output mscoco.test
```

### 2. BERT Baseline

**[Finetuning once]** on both SNLI and MNLI
```
mkdir models

GPUID=[GPUID]
LR=0.00003
PERC=1
for SEED in `seq 1 3`; do
	CUDA_VISIBLE_DEVICES=$GPUID python3 -u train.py --gpuid 0 --bert_gpuid 0 --bert_type roberta-base --dir ./data/bert_nli/ \
	--train_data mnli.train.hdf5 --val_data mnli.val.hdf5 --extra_train_data snli.train.hdf5 --extra_val_data snli.val.hdf5 \
	--learning_rate $LR --epochs 3 --warmup_epoch 3 \
	--enc bert --cls linear --hidden_size 768 --percent $PERC --dropout 0.0 \
	--fix_bert 0 --optim adam_fp16 --fp16 1 --seed ${SEED} \
	--save_file models/scratch_mnli_snli_perc${PERC//.}_seed${SEED} | tee models/scratch_mnli_snli_perc${PERC//.}_seed${SEED}.txt
done
```
Change ```[GPUID]``` to the desired device id, ```PERC``` specifies percentages of training data to use (1 is 100%). The above script will initiate BERT baselines with three different random seeds (i.e. three runs in a row). Expect to see exactly the same accuracy as we reported in our paper.

We also disabled the dropout in the final linear layer. However, there will be a dropout 0.1 (by default) inside of Bert during training.

**[Finetuning twice]** on both SNLI and MNLI

```
GPUID=[GPUID]
LR=0.00001
PERC=1
for SEED in `seq 1 3`; do
CUDA_VISIBLE_DEVICES=$GPUID python3 -u train.py --gpuid 0 --bert_gpuid 0 --bert_type roberta-base --dir ./data/bert_nli/ \
	--train_data mnli.train.hdf5 --val_data mnli.val.hdf5 --extra_train_data snli.train.hdf5 --extra_val_data snli.val.hdf5 \
	--learning_rate $LR --epochs 3 --warmup_epoch 3 \
	--enc bert --cls linear --hidden_size 768 --percent $PERC --dropout 0.0 \
	--fix_bert 0 --optim adam_fp16 --fp16 1 --seed ${SEED} \
	--load_file models/scratch_mnli_snli_perc${PERC//.}_seed${SEED} \
	--save_file models/twice_scratch_mnli_snli_perc${PERC//.}_seed${SEED} | tee models/twice_scratch_mnli_snli_perc${PERC//.}_seed${SEED}.txt
done
```
This will load the previously finetuned model and continue finetune with lowered learning rate. Expect to see exactly the same accuracy as we reported in our paper.


**[Evaluation]** on SNLI test set
```
GPUID=[GPUID]
PERC=1
SEED=[SEED]
CUDA_VISIBLE_DEVICES=$GPUID python3 -u eval.py --gpuid 0 --bert_gpuid 0 --bert_type roberta-base --dir data/bert_nli/ --data snli.test.hdf5 \
--enc bert --cls linear --hidden_size 768 --fp16 1 --dropout 0.0 \
--load_file models/scratch_mnli_snli_perc${PERC//.}_seed${SEED} | tee models/scratch_mnli_snli_perc${PERC//.}_seed${SEED}.evallog.txt
```
For MNLI, use ```--data mnli.test.hdf5```.


**[Evaluation]** on mirror consistency
```
GPUID=[GPUID]
PERC=1
for SWAP_SENT in 0 1; do
for SEED in `seq 1 3`; do
CUDA_VISIBLE_DEVICES=$GPUID python3 -u eval.py --gpuid 0 --bert_gpuid 0 --bert_type roberta-base --dir data/bert_nli/ --data mscoco.test.hdf5 \
	--enc bert --cls linear --hidden_size 768 --fp16 1 --dropout 0.0 --swap_sent $SWAP_SENT \
	--pred_output models/twice_scratch_mnli_snli_perc${PERC//.}_seed${SEED}_swap${SWAP_SENT} \
	--load_file models/twice_scratch_mnli_snli_perc${PERC//.}_seed${SEED} | tee models/twice_scratch_mnli_snli_perc${PERC//.}_seed${SEED}.evallog.txt
done
done
```

**[Evaluation]** on transitivity consistency
```
GPUID=[GPUID]
PERC=1
for PAIR in alpha beta gamma; do
for SEED in `seq 1 3`; do
CUDA_VISIBLE_DEVICES=$GPUID python3 -u eval.py --gpuid 0 --bert_gpuid 0 --bert_type roberta-base --dir data/bert_nli/ --data mscoco.test.hdf5 \
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
	CUDA_VISIBLE_DEVICES=$GPUID python3 -u train.py --gpuid 0 --bert_gpuid 0 --bert_type roberta-base --dir ./data/bert_nli/ \
	--train_data mnli.train.hdf5 --val_data mnli.val.hdf5 --extra_train_data snli.train.hdf5 --extra_val_data snli.val.hdf5 \
	--learning_rate $LR --epochs 3 --warmup_epoch 3 \
	--loss transition --fwd_mode flip --lambd ${LAMBD} \
	--enc bert --cls linear --hidden_size 768 --percent $PERC --dropout 0.0 --constr ${CONSTR} \
	--fix_bert 0 --optim adam_fp16 --fp16 1 --seed ${SEED} \
	--load_file models/scratch_mnli_snli_perc${PERC//.}_seed${SEED} \
	--save_file models/both_flip${CONSTR//,}_lr${LR//.}_lambd${LAMBD//.}_perc${PERC//.}_seed${SEED} | tee models/both_flip${CONSTR//,}_lr${LR//.}_lambd${LAMBD//.}_perc${PERC//.}_seed${SEED}.txt
done
```
Do change ```PERC``` and ```LAMBD``` accordingly.

**[Evaluation]** on mirror consistency
```
GPUID=[GPUID]
LR=0.00001
CONSTR=6
PERC=0.2
LAMBD=1
for SWAP_SENT in 0 1; do
for SEED in `seq 1 3`; do
	CUDA_VISIBLE_DEVICES=$GPUID python3 -u eval.py --gpuid 0 --bert_gpuid 0 --bert_type roberta-base --dir ./data/bert_nli/ --data mscoco.test.hdf5 \
	--enc bert --cls linear --dropout 0.0 --hidden_size 768 --fp16 1 --data_triple_mode 0 --swap_sent $SWAP_SENT \
	--pred_output models/both_flip${CONSTR//,}_lr${LR//.}_lambd${LAMBD//.}_perc${PERC//.}_seed${SEED}_swap${SWAP_SENT} \
	--load_file models/both_flip${CONSTR//,}_lr${LR//.}_lambd${LAMBD//.}_perc${PERC//.}_seed${SEED} | tee models/both_flip${CONSTR//,}_lr${LR//.}_lambd${LAMBD//.}_perc${PERC//.}_seed${SEED}.triplelog.txt
done
done

python3 confusion_table.py --log both_flip${CONSTR//,}_lr${LR//.}_lambd${LAMBD//.}_perc${PERC//.}
```

**[Evaluation]** on transitivity consistency
```
GPUID=[GPUID]
LR=0.00001
CONSTR=6
PERC=0.2
LAMBD=1
for PAIR in alpha beta gamma; do
for SEED in `seq 1 3`; do
	CUDA_VISIBLE_DEVICES=$GPUID python3 -u eval.py --gpuid 0 --bert_gpuid 0 --bert_type roberta-base --dir ./data/bert_nli/ --data mscoco.test.hdf5 \
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
CUDA_VISIBLE_DEVICES=$GPUID python3 -u train.py --gpuid 0 --bert_gpuid 0 --bert_type roberta-base --dir ./data/bert_nli/ \
	--train_data mnli.train.hdf5 --val_data mnli.val.hdf5 --extra_train_data snli.train.hdf5 --extra_val_data snli.val.hdf5 \
	--unlabeled_data mscoco.hdf5 --unlabeled_triple_mode 0 \
	--loss transition --fwd_mode flip_and_unlabeled --lambd ${LAMBD} \
	--learning_rate $LR --epochs 3 --warmup_epoch 3 --dropout 0.0 --constr ${CONSTR} \
	--enc bert --cls linear --hidden_size 768 --percent $PERC --unlabeled_perc ${PERC_U} --lambd_p $LAMBD_P \
	--fix_bert 0 --optim adam_fp16 --fp16 1 --seed ${SEED} \
	--load_file models/scratch_mnli_snli_perc${PERC//.}_seed${SEED} \
	--save_file models/both_mscoco_flip${CONSTR//,}_lr${LR//.}_lambd${LAMBD//.}_${LAMBD_P//.}_perc${PERC//.}_${PERC_U//.}_seed${SEED} | tee models/both_mscoco_flip${CONSTR//,}_lr${LR//.}_lambd${LAMBD//.}_${LAMBD_P//.}_perc${PERC//.}_${PERC_U//.}_seed${SEED}.txt
done
```
Here we set ```PERC_U=0.25``` to sample about ```100k``` unlabeled instance pairs(U) for training.

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
CUDA_VISIBLE_DEVICES=$GPUID python3 -u train.py --gpuid 0 --bert_gpuid 0 --bert_type roberta-base --dir ./data/bert_nli/ \
	--train_data mnli.train.hdf5 --val_data mnli.val.hdf5 --extra_train_data snli.train.hdf5 --extra_val_data snli.val.hdf5 \
	--unlabeled_data mscoco.hdf5 --unlabeled_triple_mode 1 \
	--loss transition --fwd_mode flip_and_triple --fix_bert 0 --optim adam_fp16 --fp16 1 --weight_decay 1 \
	--learning_rate $LR --epochs 3 --warmup_epoch 3 --dropout 0.0 --constr ${CONSTR} \
	--enc bert --cls linear --hidden_size 768 --percent $PERC --unlabeled_perc ${PERC_U} --lambd ${LAMBD} --lambd_p $LAMBD_P --lambd_t $LAMBD_T \
	--seed ${SEED} \
	--load_file models/scratch_mnli_snli_perc${PERC//.}_seed${SEED} \
	--save_file models/both_mscoco_flip_triple${CONSTR//,}_lr${LR//.}_lambd${LAMBD//.}_${LAMBD_P//.}_${LAMBD_T//.}_perc${PERC//.}_${PERC_U//.}_seed${SEED} | tee models/both_mscoco_flip_triple${CONSTR//,}_lr${LR//.}_lambd${LAMBD//.}_${LAMBD_P//.}_${LAMBD_T//.}_perc${PERC//.}_${PERC_U//.}_seed${SEED}.txt
done

Here we set ```PERC_U=0.25``` to sample about ```100k``` unlabeled instance triples(T) for training.

```
Do change ```PERC```, ```LAMBD```, and ```LAMBD_P``` accordingly. For evaluation, construct evaluation script accordingly as above.

## Hyperparameters

Please refer to the appendices of our paper for details of hyperparameters. The ``--learning_rate``, ``--lambd``, ``--lambd_p``, and ``--lambd_t`` change over different percentages ``--percent`` and ``--unlabeled_perc``.

## Issues & To-dos
- [x] Sanity check
- [ ] Cleanup code on glove and char embeddings
