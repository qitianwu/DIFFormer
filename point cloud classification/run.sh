# ActsTrack
python main.py --dataset actstrack --batch_size 1024 --lr 0.0015  \
    --method difformer --num_layers 2 --hidden_channels 64 --dropout 0.4 --weight_decay 1e-3 \
    --graph_pooling mean --kernel simple --use_bn --use_residual --use_graph --use_weight \
    --device 0  --runs 3 --epochs 150 --display_step 10 \
    --rand_split_class --train_prop 0.5 --valid_prop 0.25

python main.py --dataset actstrack --batch_size 1024 --lr 0.0015  \
    --method difformer --num_layers 2 --hidden_channels 64 --dropout 0.4 --weight_decay 1e-3 \
    --graph_pooling mean --kernel sigmoid --use_bn --use_residual --use_graph --use_weight \
    --device 0  --runs 3 --epochs 150 --display_step 10 \
    --rand_split_class --train_prop 0.5 --valid_prop 0.25 

# Tau3Mu
python main.py --dataset tau3mu --batch_size 8192 --lr 0.015 \
    --method difformer --num_layers 2 --hidden_channels 64 --dropout 0.3 --weight_decay 5e-05 \
    --graph_pooling mean --kernel simple --use_bn --use_residual --use_graph --use_weight \
    --device 3 --runs 3 --epochs 100 --display_step 10 \
    --rand_split_class --train_prop 0.5 --valid_prop 0.25

python main.py --dataset tau3mu --batch_size 8192 --lr 0.005 \
    --method difformer --num_layers 2 --hidden_channels 64 --dropout 0.3 --weight_decay 5e-05 \
    --graph_pooling mean --kernel simple --use_bn --use_residual --use_graph --use_weight \
    --device 3 --runs 3 --epochs 100 --display_step 10 \
    --rand_split_class --train_prop 0.5 --valid_prop 0.25

# SynMol
python main.py --dataset synmol --batch_size 8192 --lr 0.01 \
    --method difformer --num_layers 2 --hidden_channels 64 --dropout 0.3 --weight_decay 1e-4 \
    --graph_pooling mean --kernel simple --use_bn --use_residual --use_graph --use_weight --alpha 0.3 \
    --device 0 --runs 3 --epochs 150 --display_step 10 \
    --rand_split_class --train_prop 0.5 --valid_prop 0.25

python main.py --dataset synmol --batch_size 8192 --lr 0.01 \
    --method difformer --num_layers 2 --hidden_channels 64 --dropout 0.4 --weight_decay 1e-4 \
    --graph_pooling mean --kernel sigmoid --use_bn --use_residual --use_graph --use_weight --alpha 0.5 \
    --device 0 --runs 3 --epochs 150 --display_step 10 \
    --rand_split_class --train_prop 0.5 --valid_prop 0.25
