# DROP
We built our project based on the source code of [Numnet](https://github.com/emnlp2020qdgat/QDGAT) and [QDGAT](https://github.com/llamazing/numnet_plus), two advanced models for discrete reasoning that are both built on RoBERTa and Graph Neural Network(GNN). To further improve the performance, we modified the GNN structure by adopting heterogeneous graph transformer. We also tried different strategies of extracting RoBERTa outputs for downstream tasks.

## Cross-validation score
Average F1: 0.6656, and average EM: 0.6330

## Run

- Download processed dataset
  
  `https://hkustconnect-my.sharepoint.com/:u:/g/personal/czhengag_connect_ust_hk/EertZinzK9ZLh90XYaMM7qsBK5SDpgWUatw8fQr8QQ3T1A?e=KwoAtC`

  `tar zxvf data.tar.gz` at base directory

- Down roberta model

  `cd data/validation`
  `mkdir roberta & cd roberta`
  `wget https://huggingface.co/deepset/roberta-base-squad2/resolve/main/pytorch_model.bin`
  `wget https://huggingface.co/deepset/roberta-base-squad2/resolve/main/config.json` and remember to add `"output_hidden_states": true` in it.
  `wget https://huggingface.co/deepset/roberta-base-squad2/resolve/main/merges.txt`
  `wget https://huggingface.co/deepset/roberta-base-squad2/resolve/main/vocab.json`

- Train with simple multi-span extraction (NumNet+). **And you can see results saved in numnet.log!**

    `sh train.sh 345 5e-4 1.5e-5 5e-5 0.01 | tee numnet.log`

