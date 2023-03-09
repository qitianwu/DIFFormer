#### spatial-temporal prediction ####

## chickenpox
# DIFFormer-s w/ graph
python main.py --dataset chickenpox --method difformer --lr 0.01 --num_layers 2 --hidden_channels 4 --weight_decay 0.0 \
  --dropout 0.2 --num_heads 1 --kernel simple --use_graph --use_bn --use_residual --alpha 0.5 --device 1 --seed 123
# DIFFormer-s w/o graph
python main.py --dataset chickenpox --method difformer --lr 0.01 --num_layers 2 --hidden_channels 4 --weight_decay 0.0 \
  --dropout 0.2 --num_heads 1 --kernel simple --use_bn --use_residual --alpha 0.5 --device 1 --seed 123
# DIFFormer-a w/ graph
python main.py --dataset chickenpox --method difformer --lr 0.005 --num_layers 2 --hidden_channels 4 --weight_decay 0.0 \
  --dropout 0.2 --num_heads 1 --kernel sigmoid --use_graph --use_bn --use_residual --alpha 0.5 --device 1 --seed 123
# DIFFormer-a w/o graph
python main.py --dataset chickenpox --method difformer --lr 0.01 --num_layers 2 --hidden_channels 4 --weight_decay 0.005 \
  --dropout 0.2 --num_heads 1 --kernel sigmoid --use_bn --use_residual --alpha 0.5 --device 1 --seed 123

## covid
# DIFFormer-s w/ graph
python main.py --dataset covid --method difformer --lr 0.01 --num_layers 2 --hidden_channels 4 --weight_decay 0.0 \
  --dropout 0.2 --num_heads 1 --kernel simple --use_graph --use_bn --use_residual --alpha 0.5 --device 1 --seed 123
# DIFFormer-s w/o graph
python main.py --dataset covid --method difformer --lr 0.01 --num_layers 2 --hidden_channels 4 --weight_decay 0.005 \
  --dropout 0.2 --num_heads 1 --kernel simple --use_bn --use_residual --alpha 0.5 --device 1 --seed 123
# DIFFormer-a w/ graph
python main.py --dataset covid --method difformer --lr 0.01 --num_layers 2 --hidden_channels 4 --weight_decay 0.005 \
  --dropout 0.5 --num_heads 1 --kernel sigmoid --use_graph --use_bn --use_residual --alpha 0.5 --device 1 --seed 123
# DIFFormer-a w/o graph
python main.py --dataset covid --method difformer --lr 0.005 --num_layers 2 --hidden_channels 4 --weight_decay 0.005 \
  --dropout 0.2 --num_heads 1 --kernel sigmoid --use_bn --use_residual --alpha 0.5 --device 1 --seed 123

## wikimath
# DIFFormer-s w/ graph
python main.py --dataset wikimath --method difformer --lr 0.005 --num_layers 2 --hidden_channels 4 --weight_decay 0.0 \
  --dropout 0.0 --num_heads 1 --kernel simple --use_graph --use_bn --use_residual --alpha 0.5 --device 1 --seed 123
# DIFFormer-s w/o graph
python main.py --dataset wikimath --method difformer --lr 0.005 --num_layers 2 --hidden_channels 4 --weight_decay 0.0 \
  --dropout 0.0 --num_heads 1 --kernel simple --use_bn --use_residual --alpha 0.5 --device 1 --seed 123
# DIFFormer-a w/ graph
python main.py --dataset wikimath --method difformer --lr 0.005 --num_layers 2 --hidden_channels 4 --weight_decay 0.0 \
  --dropout 0.0 --num_heads 1 --kernel sigmoid --use_graph --use_bn --use_residual --alpha 0.5 --device 1 --seed 123
# DIFFormer-a w/o graph
python main.py --dataset wikimath --method difformer --lr 0.005 --num_layers 2 --hidden_channels 4 --weight_decay 0.0 \
  --dropout 0.0 --num_heads 1 --kernel sigmoid --use_bn --use_residual --alpha 0.5 --device 1 --seed 123