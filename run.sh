python3 main.py --expName "baseline_pre_trained6" --train --epochs 200 --netLength 6 --gpus 0 --workers 1 --taskSize 8 --batchSize 64 --weightsToKeep 10  @configs/args1.txt --earlyStopping 20 --restoreEpoch 1
python3 main.py --expName "baseline_pre_trained12" --train --epochs 200 --netLength 12 --gpus 0 --workers 1 --taskSize 8 --batchSize 64 --weightsToKeep 10  @configs/args1.txt --earlyStopping 20 --restoreEpoch 1
python3 main.py --expName "baseline_pre_trained_scratch" --train --epochs 200 --netLength 6 --gpus 0 --workers 1 --taskSize 8 --batchSize 64 --weightsToKeep 10  @configs/args1.txt --earlyStopping 20




