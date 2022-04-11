## Setup the environment

1. Set python environment:

```bash
cd feature_extracion
conda create -n feature_extraction --file requirements.txt
conda activate feature_extraction
pip install websocket websocket-client transformers moviepy
```

Install audio backend:
SoundFile for Windows `pip install PySoundFile`
Sox for Linux `pip install sox`

3. We use the gluon-cv to extract visual features. Clone it to the gluon-cv directory:
```bash
git clone git@github.com:dmlc/gluon-cv.git
```
We will use the script `gluon-cv/scripts/action-recognition/feat_extract.py`.

3. Set weights of bert:

Download the bert pre-trained weights `pytorch_model.bin` in bert directory (due to the limited size, we can't provide weights directly)
from `https://huggingface.co/bert-base-uncased/tree/main`.

## Prepare Data
Put videos and scripts in `data` directory and then config input paths in `feature_extraction.py` file.


## Running the code

```bash
python feature_extraction.py
```
The extracted feature files will be output to the same directory as input files.
