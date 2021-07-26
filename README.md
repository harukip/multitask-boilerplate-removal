# Multi-Task Neural Sequence Labeling for Zero-shot Cross-Lingual Boilerplate Removal

## requirements
- tensorflow>=2.3.0
- tqdm
- pandas
- tensorflow_hub
- sklearn
- beautifulsoup4
  - with [lxml](https://lxml.de/) library

## Basic Usage
Default Setting of MultiBoilerNet (Tag Vector + BERT + Depth Prediction)
train and test on Cleaneval EN.
```
python main.py
```

## Preprocessing
Preprocess the custom html/txt file into csv files.
```
python preprocess.py --input_path="/path/to/your/html/folder/" --output_path="/path/to/your/output/folder/" --file_type="html"
```
You can switch the file type by changing `file_type` from "html" to "txt" if your input is txt file.

### Process file with label
If the input raw file had been annotated with labels using attribute called `__boilernet_label` in HTML Tags, you can add argument in preprocess.py `--with_label` to generate csvs with labels.
Example:
```
python preprocess.py --input_path="/path/to/your/html/folder/" --output_path="/path/to/your/output/folder/" --file_type="html" --with_label
```

## Arguments
In main.py there exists several arguments that can adjust by yourself.

- `--bayesian=0` disable bayesian LSTM and use normal LSTM instead (default is 1).
- `--batch=NUM` set the training batch size to NUM (default is 1).
- `--epoch=NUM` set the training epochs to NUM (default is 20).
- `--alpha=NUM` set the auxilary task factor to NUM (default is 0.5).
- `--aux=NUM` set auxiliary task (0 for none, 1 for depth, 2 for pos) (default is 1).
- `--tag_rep` set tag representation (0 for vector, 1 for embedding) (default is 0).
- `--emb_init` set tag representation (0 for random, 1 for cbow, 2 for skip-gram) (default is 2).
- `--train_folder` set train csvs location (default="./data/cleaneval/train/").
- `--val_folder` set val csvs location (default="./data/cleaneval/val/").
- `--test_folder` set test csvs location (default="./data/cleaneval/test/").
- `--train` **Remember to set this if you want to train.** if not set, model will use the latest checkpoint and will not train new model (default is False)
- `--verbose=NUM` set the output details while training (0 for not showing losses, 1 for print all) (default is 1).

Example Usage:

(Tag Embedding + BERT + w/o Auxiliary Task)
```
python main.py --train --tag_rep=1 --aux=0
```
