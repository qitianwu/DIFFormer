datasets=('cora' 'citeseer' 'pubmed')
dev=3

for d in ${datasets[@]}
  do
  for a in 0 0.0001 0.001 0.01 0.1 1.
  do
    for b in 0. 0.2 0.5
    do
      for c in 16 32 64
      do
        for e in 2 4 8 16
        do
          python main.py --dataset $d --method difformer --rand_split_class --lr 0.001 --num_layers $e --hidden_channels $c --weight_decay $a --dropout $b --num_heads 1 --kernel simple --use_graph --use_bn --use_residual --alpha 0.5 --device $dev --seed 123 --save_result
          python main.py --dataset $d --method difformer --rand_split_class --lr 0.001 --num_layers $e --hidden_channels $c --weight_decay $a --dropout $b --num_heads 1 --kernel sigmoid --use_graph --use_bn --use_residual --alpha 0.5 --device $dev --seed 123 --save_result
        done
      done
    done
  done
done

