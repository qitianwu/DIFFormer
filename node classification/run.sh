# small graph datasets, standard random splits, 140/500/1000
python main.py --dataset cora --method difformer --rand_split_class --lr 0.001 --weight_decay 0.01 --dropout 0.2 --num_layers 8 --hidden_channels 64 --num_heads 1 --kernel simple --use_graph --use_bn --use_residual --alpha 0.5 --runs 5 --epochs 500 --seed 123 --device 2
python main.py --dataset cora --method difformer --rand_split_class --lr 0.001 --weight_decay 0.1 --dropout 0.0 --num_layers 8 --hidden_channels 64 --num_heads 1 --kernel sigmoid --use_graph --use_bn --use_residual --alpha 0.5 --runs 5 --epochs 500 --seed 123 --device 2

python main.py --dataset citeseer --method difformer --rand_split_class --lr 0.001 --weight_decay 0.1 --dropout 0.0 --num_layers 8 --hidden_channels 64 --num_heads 1 --kernel simple --use_graph --use_bn --use_residual --alpha 0.5 --runs 5 --epochs 500 --seed 123 --device 1
python main.py --dataset citeseer --method difformer --rand_split_class --lr 0.001 --weight_decay 0.1 --dropout 0.0 --num_layers 8 --hidden_channels 32 --num_heads 1 --kernel sigmoid --use_graph --use_bn --use_residual --alpha 0.5 --runs 5 --epochs 500 --seed 123 --device 1

python main.py --dataset pubmed --method difformer --rand_split_class --lr 0.001 --weight_decay 0.01 --dropout 0.0 --num_layers 8 --hidden_channels 64 --num_heads 1 --kernel simple --use_graph --use_bn --use_residual --alpha 0.5 --runs 5 --epochs 500 --seed 123 --device 1
python main.py --dataset pubmed --method difformer --rand_split_class --lr 0.001 --weight_decay 0.01 --dropout 0.0 --num_layers 8 --hidden_channels 64 --num_heads 1 --kernel sigmoid --use_graph --use_bn --use_residual --alpha 0.5 --runs 5 --epochs 500 --seed 123 --device 1

# large graph datasets
python main-batch.py --dataset ogbn-proteins --method difformer --metric rocauc --lr 1e-2 --weight_decay 0. --num_layers 3 --hidden_channels 64 --num_heads 1 --kernel simple --use_graph --use_bn --use_residual --use_weight --batch_size 10000 --runs 5 --epochs 1000 --seed 123 --device 1
python main-batch.py --dataset pokec --method difformer --rand_split --metric acc --lr 1e-2 --weight_decay 0. --num_layers 3 --hidden_channels 128 --num_heads 1 --kernel simple --use_graph --use_bn --use_residual --use_weight --batch_size 100000 --runs 5 --epochs 500 --seed 123 --device 3

