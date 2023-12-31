# Searching for optimal architecture for ResNet or ConvNet using Optuna grid search for 1d data


The repo helps you search for a optimal architecture for a given dataset Convolutional Neural Network or Residual Neural Network for 1d signals. 
The general architecture is each ConvNet/ResNet is divided into stages where features are kept fix at each one. Several layers at each stage increasing sequentially. For each trail either ConvNet or ResNet with given stages, layers and features at each stage is selected. In addition hyperparamters is also searched for such as learning rate and batch size. For example a two stage ResNet with two and four layers at each stage with 32 and 64 features/filters at each stage is shown as  
** ResNet [2 4] [32 64] **

# Schematic of general architecture

![schematic](schematic.png)

# Dataset

You need to provide a dataset file which is in .npy format for the code to work. All array has the shape  Examples x Size of each example. The input and corresponding outputs are xdata and ydata respectively. Each sample is weighted, if not please provide  
```numpy.ones(xdata.shape)``` as array.

The dataset file is read as
```python

    with open(dataset_file, 'rb') as f:
        xdata = np.load(f)
        ydata = np.load(f)
        weights = np.load(f)
```


# Applying  changes to grid search

To make chnages to the laid out grid search for architecture and hyper paramters please modify this part of section in file objective_func.py

```python
    # choose hyper parameters for model training
    initial_learning_rate = trial.suggest_float('lr', 1e-4, 0.5, log=True)
    # powers of two
    batch_size = np.power(2, trial.suggest_int("batch_size", 5, 10))
    # four stages, inspired by ResNet
    num_blocks = np.full(trial.suggest_int("num_blocks", 1, 2), 1, dtype=np.int32)
    
    features_per_block = np.ones(num_blocks.shape, dtype=np.int32)
    layers_per_block = np.ones(num_blocks.shape, dtype=np.int32)
    # the features at first stage keeping feature as power of two
    features_per_block[0] = np.power(2, trial.suggest_int("features_per_block1", 2, 5))
    layers_per_block[0] = trial.suggest_int("num_layers1", 1, 2)
    
    # layers at each stage, maximum
    max_layers_per_block = 4

    for ci in range(1,len(num_blocks)):
        features_per_block[ci] = trial.suggest_int("features_per_block"+str(ci+1), 
                                             features_per_block[ci-1], 
                                             features_per_block[ci-1]*2, 
                                             features_per_block[ci-1])

        #each stage choice between number of layers in 
        #previous stage and up to maximum value
        layers_per_block[ci] = trial.suggest_int("layers_per_block"+str(ci+1), 
                                             layers_per_block[ci-1], 
                                             max_layers_per_block)

    
    # suggest a network
    network = trial.suggest_categorical("network", ["ConvNet", "ResNet"])
```


# Usages
```command
main.py [-h] [--epochs EPOCHS] [--patience_epochs PATIENCE_EPOCHS] [--trails TRAILS] [--seed SEED] [--load_study LOAD_STUDY] [--dataset_file DATASET_FILE] [--study_file STUDY_FILE]
```

# Output for the command
 ```command
 python main.py --load_study False --study_file study.pkl  --epochs 20 --trails 10
```

```command

epochs, trails,  patience_epochs =  20 10 10
created study file
[I 2023-11-07 19:22:22,470] A new study created in memory with name: no-name-bdd2786c-0b59-4ebc-90c2-c78fb116f6a7
dataset.npy

ResNet [2 4] [32 64]

number of batches = 7.8125

Epoch 1 15.2 [sec] 30487.012000 0.415764 0
......
Epoch 20 6.8 [sec] 52.063090 0.290753 3


[I 2023-11-07 19:24:47,600] Trial 0 finished with value: 43647.39453125 and parameters: {'lr': 0.07641435988290146, 'batch_size': 7, 'num_blocks': 2, 'features_per_block1': 5, 'num_layers1': 2, 'features_per_block2': 64, 'layers_per_block2': 4, 'network': 'ResNet'}. Best is trial 0 with value: 43647.39453125.

trail number = 1
Best value: 43647.39453125, Best params: {'lr': 0.07641435988290146, 'batch_size': 7, 'num_blocks': 2, 'features_per_block1': 5, 'num_layers1': 2, 'features_per_block2': 64, 'layers_per_block2': 4, 'network': 'ResNet'}
saving study in to file study.pkl



ResNet [2 3] [32 64]
number of batches = 1.953125
Epoch 1 11.0 [sec] 6274357919.744000 0.886518 0
........

[I 2023-11-07 19:26:37,707] Trial 1 finished with value: 14073.416015625 and parameters: {'lr': 0.029168704673281753, 'batch_size': 9, 'num_blocks': 2, 'features_per_block1': 5, 'num_layers1': 2, 'features_per_block2': 64, 'layers_per_block2': 3, 'network': 'ResNet'}. Best is trial 1 with value: 14073.416015625

trail number = 2
Best value: 14073.416015625, Best params: {'lr': 0.029168704673281753, 'batch_size': 9, 'num_blocks': 2, 'features_per_block1': 5, 'num_layers1': 2, 'features_per_block2': 64, 'layers_per_block2': 3, 'network': 'ResNet'}<br>
saving study in to file study.pkl


ConvNet [2] [8]
number of batches = 1.953125
Epoch 1 3.1 [sec] 203614920376.320007 1.379745 0
.........

[I 2023-11-07 19:27:02,401] Trial 2 finished with value: 117902.59375 and parameters: {'lr': 0.005381721772891036, 'batch_size': 9, 'num_blocks': 1, 'features_per_block1': 3, 'num_layers1': 2, 'network': 'ConvNet'}. Best is trial 1 with value: 14073.416015625

trail number = 3
Best value: 14073.416015625, Best params: {'lr': 0.029168704673281753, 'batch_size': 9, 'num_blocks': 2, 'features_per_block1': 5, 'num_layers1': 2, 'features_per_block2': 64, 'layers_per_block2': 3, 'network': 'ResNet'}<br>
saving study in to file study.pkl


ConvNet [2 4] [16 32]
number of batches = 7.8125

Epoch 1 6.9 [sec] 2693463801.856000 0.641244 0
..........
Epoch 20 2.8 [sec] 2.651378 0.350648 0

[I 2023-11-07 19:28:02,504] Trial 3 finished with value: 42422.05078125 and parameters: {'lr': 0.004493857248617635, 'batch_size': 7, 'num_blocks': 2, 'features_per_block1': 4, 'num_layers1': 2, 'features_per_block2': 32, 'layers_per_block2': 4, 'network': 'ConvNet'}. Best is trial 1 with value: 14073.416015625

trail number = 4
Best value: 14073.416015625, Best params: {'lr': 0.029168704673281753, 'batch_size': 9, 'num_blocks': 2, 'features_per_block1': 5, 'num_layers1': 2, 'features_per_block2': 64, 'layers_per_block2': 3, 'network': 'ResNet'}<br>
saving study in to file study.pkl


ResNet [1 4] [16 32]
number of batches = 31.25
Epoch 1 12.1 [sec] 16112.469000 0.503134 0

[I 2023-11-07 19:30:19,675] Trial 4 finished with value: 125646.375 and parameters: {'lr': 0.0015244371181319049, 'batch_size': 5, 'num_blocks': 2, 'features_per_block1': 4, 'num_layers1': 1, 'features_per_block2': 32, 'layers_per_block2': 4, 'network': 'ResNet'}. Best is trial 1 with value: 14073.416015625

trail number = 5
Best value: 14073.416015625, Best params: {'lr': 0.029168704673281753, 'batch_size': 9, 'num_blocks': 2, 'features_per_block1': 5, 'num_layers1': 2, 'features_per_block2': 64, 'layers_per_block2': 3, 'network': 'ResNet'}<br>
saving study in to file study.pkl


ConvNet [1] [32]
number of batches = 1.953125
Epoch 1 3.3 [sec] 1099948573589.503906 2.447904 0
.............
Epoch 20 2.4 [sec] 5.607065 0.424946 0

[I 2023-11-07 19:31:08,217] Trial 5 finished with value: 22428.26171875 and parameters: {'lr': 0.015774233357113494, 'batch_size': 9, 'num_blocks': 1, 'features_per_block1': 5, 'num_layers1': 1, 'network': 'ConvNet'}. Best is trial 1 with value: 14073.416015625

trail number = 6
Best value: 14073.416015625, Best params: {'lr': 0.029168704673281753, 'batch_size': 9, 'num_blocks': 2, 'features_per_block1': 5, 'num_layers1': 2, 'features_per_block2': 64, 'layers_per_block2': 3, 'network': 'ResNet'}<br>
saving study in to file study.pkl


ConvNet [1 4] [16 16]
number of batches = 0.9765625
Epoch 1 4.5 [sec] 54329.084000 0.462086 0
................
Epoch 20 1.4 [sec] 3.714748 0.429612 0

[I 2023-11-07 19:31:39,124] Trial 6 finished with value: 7429.49658203125 and parameters: {'lr': 0.0002775059131499799, 'batch_size': 10, 'num_blocks': 2, 'features_per_block1': 4, 'num_layers1': 1,<br> 'features_per_block2': 16, 'layers_per_block2': 4, 'network': 'ConvNet'}. Best is trial 6 with value: 7429.49658203125

trail number = 7
Best value: 7429.49658203125, Best params: {'lr': 0.0002775059131499799, 'batch_size': 10, 'num_blocks': 2, 'features_per_block1': 4, 'num_layers1': 1, 'features_per_block2': 16, 'layers_per_block2': 4, 'network': 'ConvNet'}<br>
saving study in to file study.pkl


ResNet [2] [4]
number of batches = 7.8125
Epoch 1 4.6 [sec] 4482.470500 0.378300 0
..............
Epoch 20 0.9 [sec] 30.107635 0.372391 5

[I 2023-11-07 19:32:02,715] Trial 7 finished with value: 50778.3125 and parameters: {'lr': 0.21320972196439056, 'batch_size': 7, 'num_blocks': 1, 'features_per_block1': 2, 'num_layers1': 2, 'network': 'ResNet'}. Best is trial 6 with value: 7429.49658203125.<br>

trail number = 8
Best value: 7429.49658203125, Best params: {'lr': 0.0002775059131499799, 'batch_size': 10, 'num_blocks': 2, 'features_per_block1': 4, 'num_layers1': 1, 'features_per_block2': 16, 'layers_per_block2': 4, 'network': 'ConvNet'}
saving study in to file study.pkl


ConvNet [2 2] [16 32]
number of batches = 7.8125
Epoch 1 6.1 [sec] 160320744062.976013 8.803352 0
...................
Epoch 15 2.5 [sec] 63443017.728000 0.340874 9

[I 2023-11-07 19:32:46,695] Trial 8 finished with value: 49819164672.0 and parameters: {'lr': 0.03179689199761305, 'batch_size': 7, 'num_blocks': 2, 'features_per_block1': 4, 'num_layers1': 2, 'features_per_block2': 32, 'layers_per_block2': 2, 'network': 'ConvNet'}. Best is trial 6 with value: 7429.49658203125

trail number = 9
Best value: 7429.49658203125, Best params: {'lr': 0.0002775059131499799, 'batch_size': 10, 'num_blocks': 2, 'features_per_block1': 4, 'num_layers1': 1, 'features_per_block2': 16, 'layers_per_block2': 4, 'network': 'ConvNet'}
saving study in to file study.pkl


ResNet [1] [8]
number of batches = 0.9765625
Epoch 1 2.6 [sec] 135388677013.503998 1.203094 0
...................
Epoch 20 0.7 [sec] 5.386303 0.447556 0

[I 2023-11-07 19:33:03,254] Trial 9 finished with value: 10772.6064453125 and parameters: {'lr': 0.03482059722810703, 'batch_size': 10, 'num_blocks': 1, 'features_per_block1': 3, 'num_layers1': 1, 'network': 'ResNet'}. Best is trial 6 with value: 7429.49658203125.

trail number = 10
Best value: 7429.49658203125, Best params: {'lr': 0.0002775059131499799, 'batch_size': 10, 'num_blocks': 2, 'features_per_block1': 4, 'num_layers1': 1, 'features_per_block2': 16, 'layers_per_block2': 4, 'network': 'ConvNet'}
saving study in to file study.pkl

Number of finished trials: 10

Best trial:
  Value: 7429.49658203125
  Params: 
lr: 0.0002775059131499799
batch_size: 10
num_blocks: 2
features_per_block1: 4
num_layers1: 1
features_per_block2: 16
layers_per_block2: 4
network: ConvNet

```



 
