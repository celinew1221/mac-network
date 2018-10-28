# Compostional Attention Networks with Action Identification for Machine Reasoning

This code is in development to solve action identification mentioned in this [work]. It is developed based on  MacNet[Compositional Attention Networks for Machine Reasoning](https://arxiv.org/pdf/1803.03067.pdf)(ICLR 2018). MacNet a fully differentiable model that learns to perform multi-step reasoning and explore it in the context of the [CLEVR dataset](http://cs.stanford.edu/people/jcjohns/clevr/). This code is meant to explore in the context of the [action-agumented CLEVR_dataset](). See the github folder for more details.

In particular, two methods are explored in this work. Firstly, a naive approach is implemented in baseline branch - where the two images are stacked together and the stacked vectors are fed into the MacNet. A second approch is to redesign the MAC cell at [`mac_cell.py`](mac_cell.py) in `intraActionCell` branch. Details of the design is shown in the images below. Images are fed separately into the network.
Run `python main.py -h` or see [`config.py`](config.py) for the complete list of options. To run action-based model, see `--incluAction`, `--actionOnlyTrain` and `--alterActionTrain`. 

<div align="center">
  <img src="https://preview.ibb.co/eKOrRU/cell.png" style="float:left" width="390px">
  <img src="https://preview.ibb.co/j0nwt9/visual.png" style="float:right" width="480px">
</div>

## Requirements
- Tensorflow (originally has been developed with 1.3 but should work for later versions as well).
- We have performed experiments on Maxwell Titan X GPU. We assume 12GB of GPU memory.
- See [`requirements.txt`](requirements.txt) for the required python packages and run `pip install -r requirements.txt` to install them.

## Pre-processing
Before training the model, we first have to download the CLEVR dataset and extract features for the images:

### Dataset
To download and unpack the data, run the following commands, where * is `v1.0` for regular CLEVR dataset and `action` for action-based CLEVR dataset. Use the following code to download CLEVR dataset and prepare it for training:
```bash
wget https://s3-us-west-1.amazonaws.com/clevr/CLEVR_*.zip
unzip CLEVR_*.zip
mv CLEVR_* CLEVR_v1
mkdir CLEVR_v1/data
mv CLEVR_v1/questions/* CLEVR_v1/data/
```
You can find action-based data here: ... dropbox link
The final command moves the dataset questions into the `data` directory, where we will put all the data files we use during training.

### Feature extraction
Extract ResNet-101 features for the CLEVR train, val, and test images with the following commands:
To extract data for intraActionCell, use argument `--mode action_sep`. To extract data for stacked images, use argument `--mode stack_action`.

```bash
python extract_features.py --input_image_dir CLEVR_v1/images/train --output_h5_file CLEVR_v1/data/train.h5
python extract_features.py --input_image_dir CLEVR_v1/images/val --output_h5_file CLEVR_v1/data/val.h5
python extract_features.py --input_image_dir CLEVR_v1/images/test --output_h5_file CLEVR_v1/data/test.h5
```
# Regular CLEVR Dataset
## Training 
To train the model, run the following command:
```bash
python main.py --expName "clevrExperiment" --train --testedNum 10000 --epochs 25 --netLength 16 @configs/args.txt
```

First, the program preprocesses the CLEVR questions. It tokenizes them and maps them to integers to prepare them for the network. It then stores a JSON with that information about them as well as word-to-integer dictionaries in the `./CLEVR_v1/data` directory.

Then, the program trains the model. Weights are saved by default to `./weights/{expName}` and statistics about the training are collected in `./results/{expName}`, where `expName` is the name we choose to give to the current experiment. 

### Notes
- The number of examples used for training and evaluation can be set by `--trainedNum` and `--testedNum` respectively.
- You can use the `-r` flag to restore and continue training a previously pre-trained model. 
- We recommend you to try out varying the number of MAC cells used in the network through the `--netLength` option to explore different lengths of reasoning processes.
- Good lengths for CLEVR are in the range of 4-16 (using more cells tends to converge faster and achieves a bit higher accuracy, while lower number of cells usually results in more easily interpretable attention maps). 

### Model variants
We have explored several variants of our model. We provide a few examples in `configs/args1-4.txt`. For instance, you can run the first by: 
```bash
python main.py --expName "experiment1" --train --testedNum 10000 --epochs 25 --netLength 6 @configs/args1.txt
```
- [`args1`](config/args1.txt) is the standard recurrent-control-memory cell. Leads to the most interpretable results among the configs.
- [`args2`](config/args2.txt) uses a non-recurrent variant of the control unit that converges faster.
- [`args3`](config/args3.txt) incorporates self-attention into the write unit.
- [`args4`](config/args4.txt) adds control-based gating over the memory.

See [`config.py`](config.py) for further available options (Note that some of them are still in an experimental stage).

## Evalutation
To evaluate the trained model, and get predictions and attention maps, run the following: 
```bash
python main.py --expName "clevrExperiment" --finalTest --testedNum 10000 --netLength 16 -r --getPreds --getAtt @configs/args.txt
```
The command will restore the model we have trained, and evaluate it on the validation set. JSON files with predictions and the attention distributions resulted by running the model are saved by default to `./preds/{expName}`.

- In case you are interested in getting attention maps (`--getAtt`), and to avoid having large prediction files, we advise you to limit the number of examples evaluated to 5,000-20,000.

## Visualization
After we evaluate the model with the command above, we can visualize the attention maps generated by running:
```bash
python visualization.py --expName "clevrExperiment" --tier val 
```
(Tier can be set to `train` or `test` as well). The script supports filtering of the visualized questions by various ways. See [`visualization.py`](visualization.py) for further details.

To get more interpretable visualizations, it is highly recommended to reduce the number of cells to 4-8 (`--netLength`). Using more cells allows the network to learn more effective ways to approach the task but these tend to be less interpretable compared to a shorter networks (with less cells).  

Optionally, to make the image attention maps look a little bit nicer, you can do the following (using [imagemagick](https://www.imagemagick.org)):
```
for x in preds/clevrExperiment/*Img*.png; do magick convert $x -brightness-contrast 20x35 $x; done;
```

## Action-Based Dataset
To run different experiments use the following commands:

### Train regular as pretrained weights
`git checkout baseline`
`python3 main.py --expName "pretrained" --train --epochs 6 --netLength 6 --gpus 1 --workers 1 --taskSize 8 --batchSize 64 --weightsToKeep 10 --earlyStopping 5 --incluAction --trainedNum 350000 --testedNum 10000 @configs/args1.txt --debug`
`python main.py --expName "pretrained" --finalTest --netLength 6 -r --getPreds --getAtt @configs/args1.txt --incluAction --debug --gpus 1`

### To train based-on the pretrained weights
We need to copy pretrained file to action_only_finetuned folder first
`cp ./weights/pretrained/config-pretrained.json ./weights/action_only_finetuned/config-pretrained.csv`
`cp ./results/pretrained/results-pretrained.csv ./results/action_only_finetuned/results-action_only_finetuned.csv`

### Method 1 - Stack Image Vectors 
`python3 main.py --expName "action_only_finetuned" --train --epochs 200 --netLength 6 --gpus 1 --workers 1 --taskSize 8 --batchSize 64 --weightsToKeep 20 --earlyStopping 10 -r --incluAction --actionOnlyTrain @configs/args1.txt --debug`

`python main.py --expName "action_only_finetuned" --finalTest --netLength 6 -r --getPreds --getAtt @configs/args1.txt --incluAction --actionOnlyTrain --gpus 1 --debug`

`python visualization.py --expName "action_only_finetuned" --tier val`

### Method 2 - Train Stack Image Vectors from Scratch
`python3 main.py --expName "action_only_scratch" --train --epochs 200 --netLength 6 --gpus 0 --workers 1 --taskSize 8 --batchSize 64 --weightsToKeep 20  --earlyStopping 20  --incluAction --actionOnlyTrain @configs/args1.txt --debug`
`python main.py --expName "action_only_scratch" --finalTest --netLength 6 -r --getPreds --getAtt @configs/args1.txt --debug --incluAction --gpus 0`
`python visualization.py --expName "action_only_scratch" --tier val`

### Method 3 - The IntraAction Cell
`git checkout intraActionCell`
`python3 main.py --expName "intraActionCell" --train --epochs 200 --netLength 6 --gpus 0 --workers 1 --taskSize 8 --batchSize 64 --weightsToKeep 20  --earlyStopping 20 --dataBasedir ./CLEVR_action @configs/args1.txt`
`python main.py --expName "intraActionCell" --finalTest --netLength 6 -r --getPreds --getAtt @configs/args1.txt --gpus 0`
`python visualization.py --expName "intraActionCell" --tier val`
