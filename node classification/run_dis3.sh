datasets=('cora' 'citeseer' 'pubmed')
dev=8

for b in 0.8 0.5 0.2 0.1
do
  for e in 2 4 8 12 16 24 32 40 48 56 64
  do
    python main.py --dataset pubmed --method difformer --rand_split_class --lr 0.001 --weight_decay 0.0001 --dropout 0.2 --num_layers $e --hidden_channels 64 --num_heads 1 --kernel simple --use_graph --use_bn --use_residual --alpha $b --device $dev --runs 3 --seed 123 --save_result
  done
done

for b in 0.8 0.5 0.2 0.1
do
  for e in 2 4 8 12 16 24 32 40 48 56 64
  do
    python main.py --dataset pubmed --method difformer --rand_split_class --lr 0.001 --weight_decay 0.0001 --dropout 0.2 --num_layers $e --hidden_channels 64 --num_heads 1 --kernel simple --use_graph --use_bn --use_residual --use_source --alpha $b --device $dev --runs 3 --seed 123 --save_result
  done
done