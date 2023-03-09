datasets=('chickenpox' 'wikimath' 'covid')
dev=1

for d in ${datasets[@]}
  do
  for a in 0 0.005
  do
    for b in 0.01 0.05 0.005
    do
      for c in 0 0.2 0.5
      do
        python main.py --dataset $d --method difformer --lr $b --num_layers 2 --hidden_channels 4 --weight_decay $a --dropout $c --num_heads 1 --kernel simple --use_bn --use_residual --alpha 0.5 --device $dev --seed 123 --save_result
        python main.py --dataset $d --method difformer --lr $b --num_layers 2 --hidden_channels 4 --weight_decay $a --dropout $c --num_heads 1 --kernel simple --use_graph --use_bn --use_residual --alpha 0.5 --device $dev --seed 123 --save_result
        python main.py --dataset $d --method difformer --lr $b --num_layers 2 --hidden_channels 4 --weight_decay $a --dropout $c --num_heads 1 --kernel sigmoid --use_bn --use_residual --alpha 0.5 --device $dev --seed 123 --save_result
        python main.py --dataset $d --method difformer --lr $b --num_layers 2 --hidden_channels 4 --weight_decay $a --dropout $c --num_heads 1 --kernel sigmoid --use_graph --use_bn --use_residual --alpha 0.5 --device $dev --seed 123 --save_result
      done
    done
  done
done
