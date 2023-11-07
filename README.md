### Searching for optimal architecture for ResNet or ConvNet using Optuna grid search for 1d data


The repo helps you search for a optimal architecture for a given dataset Convolutional Neural Network or Residual Neural Network for 1d signals. 
The general architecture is each ConvNet/ResNet is divided into stages where fearures are kept fix at each one. There several layers at each stages increasing sequentially. 

# Usages
usage: main.py [-h] [--epochs EPOCHS] [--patience_epochs PATIENCE_EPOCHS] [--trails TRAILS] [--seed SEED] [--load_study LOAD_STUDY] [--dataset_file DATASET_FILE] [--study_file STUDY_FILE]


# Output for the command
 python main.py --load_study False --study_file study.pkl  --epochs 20 --trails 10


 epochs, trails,  patience_epochs =  20 10 10
created study file
[I 2023-11-07 19:22:22,470] A new study created in memory with name: no-name-bdd2786c-0b59-4ebc-90c2-c78fb116f6a7
dataset.npy
Metal device set to: Apple M1 Pro

systemMemory: 16.00 GB
maxCacheSize: 5.33 GB

2023-11-07 19:22:22.522696: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:305] Could not identify NUMA node of platform GPU ID 0, defaulting to 0. Your kernel may not have been built with NUMA support.
2023-11-07 19:22:22.522857: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:271] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 0 MB memory) -> physical PluggableDevice (device: 0, name: METAL, pci bus id: <undefined>)


ResNet [2 4] [32 64]
number of batches = 7.8125
2023-11-07 19:22:23.401260: W tensorflow/python/util/util.cc:368] Sets are not currently considered sequences, but this may change in the future, so consider avoiding using them.
2023-11-07 19:22:24.268023: W tensorflow/core/platform/profile_utils/cpu_utils.cc:128] Failed to get CPU frequency: 0 Hz
2023-11-07 19:22:24.269368: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:112] Plugin optimizer for device_type GPU is enabled.
2023-11-07 19:22:32.884571: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:112] Plugin optimizer for device_type GPU is enabled.
2023-11-07 19:22:36.391551: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:112] Plugin optimizer for device_type GPU is enabled.
2023-11-07 19:22:37.550980: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:112] Plugin optimizer for device_type GPU is enabled.
Epoch 1 15.2 [sec] 30487.012000 0.415764 0
Epoch 2 7.0 [sec] 1519144.960000 0.340026 1
Epoch 3 7.1 [sec] 26603.620000 0.326824 0
Epoch 4 6.7 [sec] 11.709255 0.310173 0
Epoch 5 6.7 [sec] 11.628438 0.305226 0
Epoch 6 6.7 [sec] 3.894254 0.304021 0
Epoch 7 6.8 [sec] 5.434859 0.298892 1
Epoch 8 7.0 [sec] 1126658.560000 0.304861 2
Epoch 9 6.7 [sec] 3.099239 0.295664 0
Epoch 10 6.7 [sec] 2.908993 0.294584 0
Epoch 11 6.7 [sec] 4.521427 0.291933 1
Epoch 12 6.7 [sec] 1031.194125 0.292859 2
Epoch 13 6.7 [sec] 2.786365 0.289942 0
Epoch 14 6.8 [sec] 2.764447 0.290485 0
Epoch 15 6.8 [sec] 2.753260 0.288201 0
Epoch 16 6.8 [sec] 3.866322 0.289719 1
Epoch 17 6.7 [sec] 2.727962 0.288435 0
Epoch 18 6.7 [sec] 513.564844 0.287538 1
Epoch 19 6.7 [sec] 136.588203 0.288522 2
Epoch 20 6.8 [sec] 52.063090 0.290753 3
[I 2023-11-07 19:24:47,600] Trial 0 finished with value: 43647.39453125 and parameters: {'lr': 0.07641435988290146, 'batch_size': 7, 'num_blocks': 2, 'features_per_block1': 5, 'num_layers1': 2, 'features_per_block2': 64, 'layers_per_block2': 4, 'network': 'ResNet'}. Best is trial 0 with value: 43647.39453125.

trail number = 1
Best value: 43647.39453125, Best params: {'lr': 0.07641435988290146, 'batch_size': 7, 'num_blocks': 2, 'features_per_block1': 5, 'num_layers1': 2, 'features_per_block2': 64, 'layers_per_block2': 4, 'network': 'ResNet'}
saving study in to file study.pkl


ResNet [2 3] [32 64]
number of batches = 1.953125
2023-11-07 19:24:48.748016: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:112] Plugin optimizer for device_type GPU is enabled.
2023-11-07 19:24:54.532351: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:112] Plugin optimizer for device_type GPU is enabled.
2023-11-07 19:24:57.176345: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:112] Plugin optimizer for device_type GPU is enabled.
2023-11-07 19:24:57.973291: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:112] Plugin optimizer for device_type GPU is enabled.
Epoch 1 11.0 [sec] 6274357919.744000 0.886518 0
Epoch 2 5.2 [sec] 1707667.200000 0.668517 0
Epoch 3 5.1 [sec] 2651.303750 0.577849 0
Epoch 4 5.1 [sec] 108.611312 0.531925 0
Epoch 5 5.2 [sec] 24.950121 0.504649 0
Epoch 6 5.3 [sec] 11.806875 0.485044 0
Epoch 7 5.1 [sec] 7.766731 0.469894 0
Epoch 8 5.1 [sec] 6.236374 0.459067 0
Epoch 9 5.1 [sec] 5.406797 0.450790 0
Epoch 10 5.1 [sec] 4.898572 0.443644 0
Epoch 11 5.2 [sec] 4.546629 0.436855 0
Epoch 12 5.2 [sec] 4.314943 0.432125 0
Epoch 13 5.2 [sec] 4.122325 0.427180 0
Epoch 14 5.2 [sec] 3.959343 0.422212 0
Epoch 15 5.2 [sec] 3.852926 0.418603 0
Epoch 16 5.2 [sec] 3.755727 0.414540 0
Epoch 17 5.2 [sec] 3.675950 0.411081 0
Epoch 18 5.2 [sec] 3.616027 0.407964 0
Epoch 19 5.2 [sec] 3.562533 0.404698 0
Epoch 20 5.2 [sec] 3.518354 0.401611 0
[I 2023-11-07 19:26:37,707] Trial 1 finished with value: 14073.416015625 and parameters: {'lr': 0.029168704673281753, 'batch_size': 9, 'num_blocks': 2, 'features_per_block1': 5, 'num_layers1': 2, 'features_per_block2': 64, 'layers_per_block2': 3, 'network': 'ResNet'}. Best is trial 1 with value: 14073.416015625.

trail number = 2
Best value: 14073.416015625, Best params: {'lr': 0.029168704673281753, 'batch_size': 9, 'num_blocks': 2, 'features_per_block1': 5, 'num_layers1': 2, 'features_per_block2': 64, 'layers_per_block2': 3, 'network': 'ResNet'}
saving study in to file study.pkl


ConvNet [2] [8]
number of batches = 1.953125
2023-11-07 19:26:38.473897: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:112] Plugin optimizer for device_type GPU is enabled.
2023-11-07 19:26:40.065211: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:112] Plugin optimizer for device_type GPU is enabled.
2023-11-07 19:26:40.786531: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:112] Plugin optimizer for device_type GPU is enabled.
2023-11-07 19:26:41.019683: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:112] Plugin optimizer for device_type GPU is enabled.
Epoch 1 3.1 [sec] 203614920376.320007 1.379745 0
Epoch 2 1.2 [sec] 8654763851.775999 0.797552 0
Epoch 3 1.1 [sec] 810315350.016000 0.700751 0
Epoch 4 1.1 [sec] 104423743.488000 0.641144 0
Epoch 5 1.1 [sec] 9100032.000000 0.590142 0
Epoch 6 1.1 [sec] 751321.088000 0.557947 0
Epoch 7 1.1 [sec] 122630.032000 0.539659 0
Epoch 8 1.1 [sec] 26536.718000 0.522569 0
Epoch 9 1.1 [sec] 5763.684000 0.509382 0
Epoch 10 1.1 [sec] 734.452375 0.498729 0
Epoch 11 1.1 [sec] 224.357375 0.490739 0
Epoch 12 1.1 [sec] 193.627641 0.485602 0
Epoch 13 1.1 [sec] 137.990453 0.479848 0
Epoch 14 1.1 [sec] 97.889445 0.476307 0
Epoch 15 1.1 [sec] 80.635094 0.473035 0
Epoch 16 1.1 [sec] 58.916859 0.470044 0
Epoch 17 1.1 [sec] 29.475648 0.467047 0
Epoch 18 1.1 [sec] 39.484129 0.465581 1
Epoch 19 1.1 [sec] 83.822086 0.463239 2
Epoch 20 1.1 [sec] 94.527805 0.460775 3
[I 2023-11-07 19:27:02,401] Trial 2 finished with value: 117902.59375 and parameters: {'lr': 0.005381721772891036, 'batch_size': 9, 'num_blocks': 1, 'features_per_block1': 3, 'num_layers1': 2, 'network': 'ConvNet'}. Best is trial 1 with value: 14073.416015625.

trail number = 3
Best value: 14073.416015625, Best params: {'lr': 0.029168704673281753, 'batch_size': 9, 'num_blocks': 2, 'features_per_block1': 5, 'num_layers1': 2, 'features_per_block2': 64, 'layers_per_block2': 3, 'network': 'ResNet'}
saving study in to file study.pkl


ConvNet [2 4] [16 32]
number of batches = 7.8125
2023-11-07 19:27:03.146356: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:112] Plugin optimizer for device_type GPU is enabled.
2023-11-07 19:27:07.099878: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:112] Plugin optimizer for device_type GPU is enabled.
2023-11-07 19:27:08.441911: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:112] Plugin optimizer for device_type GPU is enabled.
2023-11-07 19:27:09.063526: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:112] Plugin optimizer for device_type GPU is enabled.
Epoch 1 6.9 [sec] 2693463801.856000 0.641244 0
Epoch 2 2.9 [sec] 9780988.928000 0.494777 0
Epoch 3 2.8 [sec] 148942.320000 0.462140 0
Epoch 4 2.9 [sec] 100.045414 0.435696 0
Epoch 5 2.8 [sec] 2579.698000 0.418334 1
Epoch 6 2.8 [sec] 135.069203 0.406745 2
Epoch 7 2.8 [sec] 17.828883 0.398562 0
Epoch 8 2.8 [sec] 3.748799 0.388981 0
Epoch 9 2.7 [sec] 3.071304 0.383684 0
Epoch 10 2.7 [sec] 2.986293 0.377501 0
Epoch 11 2.7 [sec] 2.927045 0.369063 0
Epoch 12 2.7 [sec] 160253.936000 0.383713 1
Epoch 13 2.8 [sec] 13.581732 0.360878 2
Epoch 14 2.7 [sec] 3.170436 0.357686 3
Epoch 15 2.8 [sec] 2.937991 0.348267 4
Epoch 16 2.8 [sec] 2.821405 0.350705 0
Epoch 17 2.8 [sec] 2.739663 0.348295 0
Epoch 18 2.7 [sec] 2.680143 0.349261 0
Epoch 19 2.7 [sec] 3.525509 0.346001 1
Epoch 20 2.8 [sec] 2.651378 0.350648 0
[I 2023-11-07 19:28:02,504] Trial 3 finished with value: 42422.05078125 and parameters: {'lr': 0.004493857248617635, 'batch_size': 7, 'num_blocks': 2, 'features_per_block1': 4, 'num_layers1': 2, 'features_per_block2': 32, 'layers_per_block2': 4, 'network': 'ConvNet'}. Best is trial 1 with value: 14073.416015625.

trail number = 4
Best value: 14073.416015625, Best params: {'lr': 0.029168704673281753, 'batch_size': 9, 'num_blocks': 2, 'features_per_block1': 5, 'num_layers1': 2, 'features_per_block2': 64, 'layers_per_block2': 3, 'network': 'ResNet'}
saving study in to file study.pkl


ResNet [1 4] [16 32]
number of batches = 31.25
2023-11-07 19:28:03.658372: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:112] Plugin optimizer for device_type GPU is enabled.
2023-11-07 19:28:12.591339: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:112] Plugin optimizer for device_type GPU is enabled.
2023-11-07 19:28:13.995080: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:112] Plugin optimizer for device_type GPU is enabled.
Epoch 1 12.1 [sec] 16112.469000 0.503134 0
Epoch 2 6.6 [sec] 30.847553 0.452171 0
Epoch 3 6.6 [sec] 4.251933 0.426040 0
Epoch 4 6.7 [sec] 3.309502 0.404211 0
Epoch 5 6.7 [sec] 3.064353 0.384331 0
Epoch 6 6.5 [sec] 2.812942 0.344109 0
Epoch 7 6.5 [sec] 2.715239 0.329308 0
Epoch 8 6.6 [sec] 2.638007 0.325788 0
Epoch 9 6.5 [sec] 2.980050 0.325315 1
Epoch 10 6.5 [sec] 2.241488 0.280687 0
Epoch 11 6.5 [sec] 2.193912 0.279125 0
Epoch 12 6.5 [sec] 2.188005 0.301565 0
Epoch 13 6.6 [sec] 2.032576 0.304197 0
Epoch 14 6.5 [sec] 1.963225 0.295256 0
Epoch 15 6.6 [sec] 2.165894 0.305676 1
Epoch 16 6.6 [sec] 4.562291 0.307710 2
Epoch 17 6.5 [sec] 19.737918 0.300487 3
Epoch 18 6.5 [sec] 2.237990 0.262062 4
Epoch 19 6.4 [sec] 4.722066 0.298406 5
Epoch 20 6.4 [sec] 5.166390 0.300397 6
[I 2023-11-07 19:30:19,675] Trial 4 finished with value: 125646.375 and parameters: {'lr': 0.0015244371181319049, 'batch_size': 5, 'num_blocks': 2, 'features_per_block1': 4, 'num_layers1': 1, 'features_per_block2': 32, 'layers_per_block2': 4, 'network': 'ResNet'}. Best is trial 1 with value: 14073.416015625.

trail number = 5
Best value: 14073.416015625, Best params: {'lr': 0.029168704673281753, 'batch_size': 9, 'num_blocks': 2, 'features_per_block1': 5, 'num_layers1': 2, 'features_per_block2': 64, 'layers_per_block2': 3, 'network': 'ResNet'}
saving study in to file study.pkl


ConvNet [1] [32]
number of batches = 1.953125
2023-11-07 19:30:20.038989: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:112] Plugin optimizer for device_type GPU is enabled.
2023-11-07 19:30:22.208924: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:112] Plugin optimizer for device_type GPU is enabled.
2023-11-07 19:30:22.728219: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:112] Plugin optimizer for device_type GPU is enabled.
2023-11-07 19:30:22.947096: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:112] Plugin optimizer for device_type GPU is enabled.
Epoch 1 3.3 [sec] 1099948573589.503906 2.447904 0
Epoch 2 2.2 [sec] 115879601766.399994 1.299525 0
Epoch 3 2.4 [sec] 7640786141.184000 0.922488 0
Epoch 4 2.4 [sec] 287835488.256000 0.753877 0
Epoch 5 2.4 [sec] 6402718.720000 0.671220 0
Epoch 6 2.4 [sec] 196344.032000 0.611233 0
Epoch 7 2.4 [sec] 14725.070000 0.571515 0
Epoch 8 2.4 [sec] 2749.961500 0.543493 0
Epoch 9 2.4 [sec] 734.099500 0.521472 0
Epoch 10 2.4 [sec] 264.820719 0.504611 0
Epoch 11 2.4 [sec] 118.573297 0.490407 0
Epoch 12 2.4 [sec] 55.714539 0.478121 0
Epoch 13 2.4 [sec] 19.513354 0.460366 0
Epoch 14 2.4 [sec] 9.204634 0.445673 0
Epoch 15 2.4 [sec] 7.579842 0.439425 0
Epoch 16 2.4 [sec] 7.094746 0.436361 0
Epoch 17 2.4 [sec] 6.674096 0.433455 0
Epoch 18 2.4 [sec] 6.196234 0.430421 0
Epoch 19 2.4 [sec] 5.889113 0.427622 0
Epoch 20 2.4 [sec] 5.607065 0.424946 0
[I 2023-11-07 19:31:08,217] Trial 5 finished with value: 22428.26171875 and parameters: {'lr': 0.015774233357113494, 'batch_size': 9, 'num_blocks': 1, 'features_per_block1': 5, 'num_layers1': 1, 'network': 'ConvNet'}. Best is trial 1 with value: 14073.416015625.

trail number = 6
Best value: 14073.416015625, Best params: {'lr': 0.029168704673281753, 'batch_size': 9, 'num_blocks': 2, 'features_per_block1': 5, 'num_layers1': 2, 'features_per_block2': 64, 'layers_per_block2': 3, 'network': 'ResNet'}
saving study in to file study.pkl


ConvNet [1 4] [16 16]
number of batches = 0.9765625
2023-11-07 19:31:08.897379: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:112] Plugin optimizer for device_type GPU is enabled.
2023-11-07 19:31:10.858449: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:112] Plugin optimizer for device_type GPU is enabled.
2023-11-07 19:31:12.484182: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:112] Plugin optimizer for device_type GPU is enabled.
Epoch 1 4.5 [sec] 54329.084000 0.462086 0
Epoch 2 1.4 [sec] 146.368766 0.446491 0
Epoch 3 1.4 [sec] 21.029887 0.440472 0
Epoch 4 1.4 [sec] 10.514685 0.437261 0
Epoch 5 1.4 [sec] 7.172941 0.435130 0
Epoch 6 1.4 [sec] 5.723292 0.433657 0
Epoch 7 1.4 [sec] 5.031180 0.432530 0
Epoch 8 1.4 [sec] 4.610508 0.431648 0
Epoch 9 1.4 [sec] 4.334766 0.430995 0
Epoch 10 1.4 [sec] 4.139920 0.430432 0
Epoch 11 1.4 [sec] 4.019305 0.430067 0
Epoch 12 1.4 [sec] 3.932024 0.429764 0
Epoch 13 1.4 [sec] 3.866289 0.429532 0
Epoch 14 1.4 [sec] 3.820001 0.429392 0
Epoch 15 1.4 [sec] 3.785663 0.429304 0
Epoch 16 1.4 [sec] 3.760060 0.429283 0
Epoch 17 1.4 [sec] 3.741652 0.429298 0
Epoch 18 1.4 [sec] 3.728887 0.429361 0
Epoch 19 1.4 [sec] 3.721096 0.429479 0
Epoch 20 1.4 [sec] 3.714748 0.429612 0
[I 2023-11-07 19:31:39,124] Trial 6 finished with value: 7429.49658203125 and parameters: {'lr': 0.0002775059131499799, 'batch_size': 10, 'num_blocks': 2, 'features_per_block1': 4, 'num_layers1': 1, 'features_per_block2': 16, 'layers_per_block2': 4, 'network': 'ConvNet'}. Best is trial 6 with value: 7429.49658203125.

trail number = 7
Best value: 7429.49658203125, Best params: {'lr': 0.0002775059131499799, 'batch_size': 10, 'num_blocks': 2, 'features_per_block1': 4, 'num_layers1': 1, 'features_per_block2': 16, 'layers_per_block2': 4, 'network': 'ConvNet'}
saving study in to file study.pkl


ResNet [2] [4]
number of batches = 7.8125
2023-11-07 19:31:39.720565: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:112] Plugin optimizer for device_type GPU is enabled.
2023-11-07 19:31:41.924720: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:112] Plugin optimizer for device_type GPU is enabled.
2023-11-07 19:31:43.042159: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:112] Plugin optimizer for device_type GPU is enabled.
2023-11-07 19:31:43.557926: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:112] Plugin optimizer for device_type GPU is enabled.
Epoch 1 4.6 [sec] 4482.470500 0.378300 0
Epoch 2 1.2 [sec] 161.045750 0.347272 0
Epoch 3 1.1 [sec] 483.145781 0.340317 1
Epoch 4 1.1 [sec] 6.934501 0.337995 0
Epoch 5 1.1 [sec] 313.405937 0.338053 1
Epoch 6 1.0 [sec] 3.217912 0.339946 0
Epoch 7 1.0 [sec] 3.755075 0.339852 1
Epoch 8 1.0 [sec] 12.518785 0.348001 2
Epoch 9 1.0 [sec] 3.254578 0.349897 3
Epoch 10 1.0 [sec] 4.324773 0.348531 4
Epoch 11 1.0 [sec] 2001.485625 0.348515 5
Epoch 12 0.9 [sec] 3.182528 0.351092 0
Epoch 13 0.9 [sec] 3.198088 0.349774 1
Epoch 14 0.9 [sec] 96.382461 0.346714 2
Epoch 15 0.9 [sec] 3.173645 0.346530 0
Epoch 16 0.9 [sec] 28.447752 0.357009 1
Epoch 17 0.9 [sec] 3.408994 0.372187 2
Epoch 18 0.9 [sec] 3.493214 0.387962 3
Epoch 19 0.9 [sec] 5.244896 0.378906 4
Epoch 20 0.9 [sec] 30.107635 0.372391 5
[I 2023-11-07 19:32:02,715] Trial 7 finished with value: 50778.3125 and parameters: {'lr': 0.21320972196439056, 'batch_size': 7, 'num_blocks': 1, 'features_per_block1': 2, 'num_layers1': 2, 'network': 'ResNet'}. Best is trial 6 with value: 7429.49658203125.

trail number = 8
Best value: 7429.49658203125, Best params: {'lr': 0.0002775059131499799, 'batch_size': 10, 'num_blocks': 2, 'features_per_block1': 4, 'num_layers1': 1, 'features_per_block2': 16, 'layers_per_block2': 4, 'network': 'ConvNet'}
saving study in to file study.pkl



 
