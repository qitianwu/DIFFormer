# reproduce results on ogbn-proteins (the public splits are provided by ogb paper)
python test_large_dataset.py --dataset ogbn-proteins --method difformer --metric rocauc --num_layers 3 --hidden_channels 64 --num_heads 1 --kernel simple --use_graph --use_bn --use_residual --use_weight --cpu

# reproduce results on amazon2m
# one needs to first download our provided fixed splits into '../data/pokec/split_0.5_0.25'
python test_large_dataset.py --dataset pokec --method difformer --metric acc --num_layers 3 --hidden_channels 128 --num_heads 1 --kernel simple --use_graph --use_bn --use_residual --use_weight --cpu

