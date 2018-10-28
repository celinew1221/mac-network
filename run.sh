#!/usr/bin/env bash

# baseline
git checkout baseline
# 1 - load weights
echo "Fined-tune and Trained only on Action Data"
python3 main.py --expName "action_only_finetuned" --train --epochs 200 --netLength 6 --gpus 0 --workers 1 --taskSize
8 --batchSize 64 --weightsToKeep 20 --earlyStopping 20 --restoreEpoch 11 --incluAction --actionOnlyTrain @configs/args1.txt
python main.py --expName "action_only_finetuned" --finalTest --netLength 6 -r --getPreds --getAtt @configs/args1.txt
python visualization.py --expName "action_only_finetuned" --tier val

# 2 - from scratch
echo "Action from Scratch"
python3 main.py --expName "action_only_scratch" --train --epochs 200 --netLength 6 --gpus 0 --workers 1 --taskSize 8
--batchSize 64 --weightsToKeep 20  --earlyStopping 20  --incluAction --actionOnlyTrain @configs/args1.txt
python main.py --expName "action_only_scratch" --finalTest --netLength 6 -r --getPreds --getAtt @configs/args1.txt
python visualization.py --expName "action_only_scratch" --tier val

# intraAction Cell
echo "intraActionCell"
git checkout intraActionCell
python3 main.py --expName "intraActionCell" --train --epochs 200 --netLength 6 --gpus 0 --workers 1 --taskSize 8
--batchSize 64 --weightsToKeep 20  --earlyStopping 20 @configs/args1.txt
python main.py --expName "intraActionCell" --finalTest --netLength 6 -r --getPreds --getAtt @configs/args1.txt
python visualization.py --expName "intraActionCell" --tier val



