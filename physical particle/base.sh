python main.py --dataset synmol --batch_size 128 --lr 0.01 \
    --method mlp --num_layers 2 --hidden_channels 64 --dropout 0.5 --weight_decay 5e-4 \
    --graph_pooling mean \
    --device 0 --runs 5


python main.py --dataset tau3mu --batch_size 128 --lr 0.01 \
    --method mlp --num_layers 2 --hidden_channels 64 --dropout 0.5 --weight_decay 5e-4 \
    --graph_pooling mean \
    --device 0 --epochs 101 --runs 5

python main.py --dataset actstrack --batch_size 128 --lr 0.01 \
    --method mlp --num_layers 2 --hidden_channels 64 --dropout 0.5 --weight_decay 5e-4 \
    --graph_pooling mean \
    --device 1 --runs 5

python main.py --dataset synmol --batch_size 128 --lr 0.01 \
    --method gat --num_layers 2 --hidden_channels 64 --dropout 0.5 --weight_decay 5e-4 \
    --graph_pooling mean \
    --device 0 --epochs 300 --runs 5

python main.py --dataset tau3mu --batch_size 128 --lr 0.01 \
    --method gat --num_layers 2 --hidden_channels 64 --dropout 0.5 --weight_decay 5e-4 \
    --graph_pooling mean \
    --device 0 --epochs 101 --runs 5

python main.py --dataset actstrack --batch_size 128 --lr 0.01 \
    --method gat --num_layers 2 --hidden_channels 64 --dropout 0.5 --weight_decay 5e-4 \
    --graph_pooling mean \
    --device 1 --runs 5
