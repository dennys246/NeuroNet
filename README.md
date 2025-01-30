# NeuroNet
This repository contains the source code for NeuroNet used for building and training AI on NeuroImaging data previously known as KleinNet. This tool can be used on any regression or classification BOLD task and is BIDS compliant to rapidly load in data and start training an AI in as little as 7 lines of code. NeuroNet at it's core is a TensorFlow wrapper that is entirely customizable to the task at hand with a tried and tested default convolutional neural network structure based off the paper (add citation). The tool incorperates modern neural network architecture's like attention mechanism's and mulit-scalar pooling to learn intricate non-learner spatial relationships in your data. Heatmaps and experimental multi-voxel pattern analysis allow for us to gain insight into the hidden layers of the model and the important patterns the model learns to predicting!

<img width="577" alt="image" src="https://github.com/user-attachments/assets/0c4b9833-34f3-438b-bdec-e50b92ee62ba" />

# How to train NeuroNet on your NeuroImaging data
To run NeuroNet first import the library and initialize a model...

```
import neuronet as nn

net = nn.NeuroNet()
```

Once TensorFlow and your default model builder has initialized, you now need to orient your model to you dataset. You can do this by calling the net.orient() function which takes three argumentsl; your BIDS compliant folder containing your neuroimaging data, your BOLD file identifier and your label file identifier. 



```
net.orient('/path/to/BIDS/folder/', '*movie*MNIPediatricAsym*preproc_bold_6mm_smoothed_deconvolved.nii', 'movie?_labels.txt')
```
NOTE: These identifiers use glob to find relavent files for each subject so you can use wildcards (* for multi-character and ? for single-character wildcards) as shown below to flexibly find data you're interested in

Once your model has been oriented to your database, it will list all subject data it found. 

From here you can customize the AI architecture and training process completely by modifying the config variable attached to your network. The config variable is intended to allow for easy saving, loading and transfer learning of your model and stores all information about your model for later reference. The tool is built so you don't need to adjust this config variable at all if you wish, but if you wish you can change any of the following variables to highly modify you AI...

```
config = {
    "dataset": "NeuroImaging Study",
    "architecture": "NeuroNet",
    "shuffle": False,
    "epochs": 10,
    "batch_size": 36,
    "negative_slope": 0.1,
    "epsilon": 1e-6,
    "learning_rate": 0.001,
    "bias": 0,
    "dropout": 0.5,
    "momentum": 0.9,
    "kernel_initializer": "glorot_uniform",
    "convolution_depth": 2,
    "init_filter_count": 8,
    "kernel_size": [2, 2, 2],
    "kernel_stride": [1, 1, 1],
    "zero_padding": "valid",
    "padding": [0, 0, 0],
    "pool_size": [2, 2, 2],
    "pool_stride": [2, 2, 2],
    "multiscale_pooling": False,
    "top_density": [2500, 1000, 400, 200, 100],
    "density_dropout": [0.2, 0.1, False, False, False],
    "outputs": [-1.0, 1.0],
    "outputs_category": ["Negative", "Positive"],
    "output_descriptor": "Outcome",
    "output_unit": "Value",
    "optimizer": "SGD",
    "use_nestrov": True,
    "use_amsgrad": True,
    "loss": "mse",
    "rebuild": False
}
```
For more information on these customization review the tutorial page or watch the tutorial video to gain more insight into how to scale the model!

From there finally we can build our model. To build the model, due to fMRI data's variable size nature, the model will load a single subjects data to figure out the dimension size. From there the tool will construct a convolutional neural network using your configuration as a framework. The model will summarize the structure once built succesfully...
```
net.built()
```

After building your network successfully, you can now load in some of your data and start training it with the respective net.load() and net.train() commands. Note you need to load at least 3 subjects per training batch. Here is an example of how to iterate through a large data sample...
```
for window in range(0, len(net.config.subject_pool), window_size): 
    net.load(count = 3, session = '0')
    net.train()
```

You can also directly pull from the config.subject_pool you generate from the net.orient() function for more custom batching...
```
for window in range(0, len(net.config.subject_pool), window_size): 
    subject_batch = net.subject_pool[window:window+window_size]
    results = net.load(subject_batch, session = '0')
    if results == None: # If wrangle failed with current batch, go to next subjects
        continue
    net.train()
```
NOTE: The net.load() function will return a False result it it failed to load enough subjects from the batch of subject you passed in to load.

