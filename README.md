# QDGAT (Question Directed Graph Attention Network for Numerical Reasoning over Text)

# Train the model
- The annotated data is in ./raw/annotated
- Pretrain the model using pretrain_split_annotated.json
- Execute the `run.sh` to run cross validation. The first parameter is the path to the data and the second one is the pretrained language model.
```
sh ./run.sh ./data/validation roberta-base
```

<!-- # Framework

<img src="qdgat_framework.jpg" alt="QDGAT Framework" style="zoom:40%;" />

# Prepare:

- Download the CoreNLP tools from https://stanfordnlp.github.io/CoreNLP/
- Using CoreNLP tools to parse drop passages and questions (You can use the properties file `corenlp.properties`)
- Run `parse_xml.py` to parse the xml from Step 1, and generate the data file for training.


# Usage:
- Execute the `run.sh` directly, which will do:
  - Parse drop data and find valid solutions for each drop question, and load data as batch.
  - Load QDGATNet model.
  - Run train and evaluate. -->
