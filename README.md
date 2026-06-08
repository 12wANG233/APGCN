# Lightweight adaptive pseudo-augmented graph convolutional network for skeleton-based gesture and action recognition

# Requirement 
```
conda create -n apgcn python==3.9
conda activate apgcn
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
pip install -e torchlight
```

# Datasets
- [SHREC'17 Track](http://www-rech.telecom-lille.fr/shrec2017-hand/)
- [DHG 14/28](http://www-rech.telecom-lille.fr/DHGdataset/)
- [NTU RGB+D 60 Skeleton](https://rose1.ntu.edu.sg/dataset/actionRecognition/)
- [NTU RGB+D 120 Skeleton](https://rose1.ntu.edu.sg/dataset/actionRecognition/)


#### Generating Data
- Generate SHREC'17 Track or DHG 14/28 dataset:
```
 cd ./data/shrec17_dataset
 # Get train data
 python gen_traindataset.py
 # Get test data
 python gen_testdataset.py
```

```
 cd ./data/DHG14-28_dataset
 # Get train and teat data
 python python gen_dhgdataset.py
```

- Generate NTU RGB+D 60 or NTU RGB+D 120 dataset:
```
 cd ./data/ntu # or cd ./data/ntu120
 # Get skeleton of each performer
 python get_raw_skes_data.py
 # Remove the bad skeleton 
 python get_raw_denoised_data.py
 # Transform the skeleton to the center of the first frame
 python seq_transformation.py
```

# Training & Testing
### Training
```
python main.py --config ./config/<the config>.yaml --device 0
```
### Testing
```
python main.py --config ./config/<the config>.yaml --work-dir <the save path> --phase test --save-score True --weights <the save path>/<the weight>.pt --device 0
```

# Acknowledgements
- Our project is based on the [Hyper-GCN](https://github.com/6UOOON9/Hyper-GCN), [RE-GCN](https://github.com/Gbouna/RE-TCN/tree/main).

Thanks to the original authors for their work!