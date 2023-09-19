## Setup the environment

```bash
conda create -n lad --file requirements.txt
conda activate lad
```

## Data Download

Download extracted features `split_data_clip.pkl` from `https://drive.google.com/file/d/1a7v6PeTogJ1x9vIOcK1ab1aLWGYaygRS/view?usp=sharing` and move it to the `data` folder.

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
@INPROCEEDINGS{10220045,
  author={Song, Wei and Wu, Bin and Zheng, Chunping and Zhang, Huayang},
  booktitle={2023 IEEE International Conference on Multimedia and Expo (ICME)}, 
  title={Detection Of Public Speaking Anxiety: A New Dataset And Algorithm}, 
  year={2023},
  volume={},
  number={},
  pages={2633-2638},
  doi={10.1109/ICME55011.2023.00448}}
```

## Contact

For any questions, please email at `songwei@bupt.edu.cn`

