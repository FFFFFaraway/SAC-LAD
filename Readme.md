## Setup the environment

```bash
conda create -n lad --file requirements.txt
conda activate lad
```

## Data Download

Download extracted features from `[...]`(Due to the email leak problem in the google drive link, we cannot provide the dataset download link under review stage) and move it to the `data` folder.

In the review stage, due to the size limit of the supplementary materials, we can only attach a small part of the dataset to verify that the code is runnable.

The dataset is saved using a python dict object, whose structure is:

```
{
  "train": {
    "v": np.array with shape(N, 2048),
    "a": np.array with shape(N, 599, 512),
    "p": np.array with shape(N, 100, 768),
    "t": np.array with shape(N, 100, 768),
    "y": np.array with shape(N, ),
  },
  "val": {...},
  "test": {...}
}
```

`v,a,p,t,y` correspond to visual, audio, script, live text and label respectively.


## Running the code

Train model using `python train.py` and test model using `python test.py`

You can edit args in these `.py` files

## Citation

If this paper is useful for your research, please cite us at:

```latex
@article{}
```

## Contact

For any questions, please email at `[...]`