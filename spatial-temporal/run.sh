#### spatial-temporal prediction ####

### chickenpox
## DIFFormer-s w/ graph
#python main.py --dataset chickenpox --method difformer --lr 0.01 --num_layers 2 --hidden_channels 4 --weight_decay 0.0 \
#  --dropout 0.2 --num_heads 1 --kernel simple --use_graph --use_bn --use_residual --alpha 0.5 --device 1 --seed 123 --save_result
## DIFFormer-s w/o graph
#python main.py --dataset chickenpox --method difformer --lr 0.01 --num_layers 2 --hidden_channels 4 --weight_decay 0.0 \
#  --dropout 0.2 --num_heads 1 --kernel simple --use_bn --use_residual --alpha 0.5 --device 1 --seed 123 --save_result
## DIFFormer-a w/ graph
#python main.py --dataset chickenpox --method difformer --lr 0.005 --num_layers 2 --hidden_channels 4 --weight_decay 0.0 \
#  --dropout 0.2 --num_heads 1 --kernel sigmoid --use_graph --use_bn --use_residual --alpha 0.5 --device 1 --seed 123 --save_result
## DIFFormer-a w/o graph
#python main.py --dataset chickenpox --method difformer --lr 0.01 --num_layers 2 --hidden_channels 4 --weight_decay 0.005 \
#  --dropout 0.2 --num_heads 1 --kernel sigmoid --use_bn --use_residual --alpha 0.5 --device 1 --seed 123 --save_result

## covid
# DIFFormer-s w/ graph
python main.py --dataset covid --method difformer --lr 0.01 --num_layers 2 --hidden_channels 4 --weight_decay 0.0 \
  --dropout 0.2 --num_heads 1 --kernel simple --use_graph --use_bn --use_residual --alpha 0.5 --device 1 --seed 123 --save_result
# DIFFormer-s w/o graph
python main.py --dataset covid --method difformer --lr 0.01 --num_layers 2 --hidden_channels 4 --weight_decay 0.005 \
  --dropout 0.2 --num_heads 1 --kernel simple --use_bn --use_residual --alpha 0.5 --device 1 --seed 123 --save_result
# DIFFormer-a w/ graph
python main.py --dataset covid --method difformer --lr 0.01 --num_layers 2 --hidden_channels 4 --weight_decay 0.005 \
  --dropout 0.5 --num_heads 1 --kernel sigmoid --use_graph --use_bn --use_residual --alpha 0.5 --device 1 --seed 123 --save_result
# DIFFormer-a w/o graph
python main.py --dataset covid --method difformer --lr 0.005 --num_layers 2 --hidden_channels 4 --weight_decay 0.005 \
  --dropout 0.2 --num_heads 1 --kernel sigmoid --use_bn --use_residual --alpha 0.5 --device 1 --seed 123 --save_result

#difformer simple: True: 0.0 0.2 0.01 0.7793 $\pm$ 0.0377
#difformer simple: False: 0.005 0.2 0.01 0.7572 $\pm$ 0.0486
#difformer sigmoid: True: 0.005 0.5 0.01 0.7790 $\pm$ 0.0287
#difformer sigmoid: False: 0.005 0.2 0.005 0.7416 $\pm$ 0.0520

## wikimath
## DIFFormer-s w/ graph
#python main.py --dataset wikimath --method difformer --lr 0.005 --num_layers 2 --hidden_channels 4 --weight_decay 0.0 \
#  --dropout 0.0 --num_heads 1 --kernel simple --use_graph --use_bn --use_residual --alpha 0.5 --device 1 --seed 123 --save_result
## DIFFormer-s w/o graph
#python main.py --dataset wikimath --method difformer --lr 0.005 --num_layers 2 --hidden_channels 4 --weight_decay 0.0 \
#  --dropout 0.0 --num_heads 1 --kernel simple --use_bn --use_residual --alpha 0.5 --device 1 --seed 123 --save_result
## DIFFormer-a w/ graph
#python main.py --dataset wikimath --method difformer --lr 0.005 --num_layers 2 --hidden_channels 4 --weight_decay 0.0 \
#  --dropout 0.0 --num_heads 1 --kernel sigmoid --use_graph --use_bn --use_residual --alpha 0.5 --device 1 --seed 123 --save_result
## DIFFormer-a w/o graph
#python main.py --dataset wikimath --method difformer --lr 0.005 --num_layers 2 --hidden_channels 4 --weight_decay 0.0 \
#  --dropout 0.0 --num_heads 1 --kernel sigmoid --use_bn --use_residual --alpha 0.5 --device 1 --seed 123 --save_result


#
#difformer simple: True: 0.0 0.2 0.01 0.9143 $\pm$ 0.0067
#difformer sigmoid: True: 0.0 0.2 0.005 0.9146 $\pm$ 0.0078
#difformer simple: False: 0.0 0.2 0.01 0.9162 $\pm$ 0.0058
#difformer sigmoid: False: 0.005 0.2 0.01 0.9155 $\pm$ 0.0056
#
#difformer simple: True: 0.0 0.2 0.01 0.7793 $\pm$ 0.0377
#difformer simple: False: 0.005 0.2 0.01 0.7572 $\pm$ 0.0486
#difformer sigmoid: True: 0.005 0.5 0.01 0.7790 $\pm$ 0.0287
#difformer sigmoid: False: 0.005 0.2 0.005 0.7416 $\pm$ 0.0520
#
#difformer simple: True: 0.0 0.0 0.005 0.7317 $\pm$ 0.0070
#difformer sigmoid: True: 0.0 0.0 0.005 0.7636 $\pm$ 0.0209
#difformer simple: False: 0.0 0.0 0.005 0.7277 $\pm$ 0.0250
#difformer sigmoid: False: 0.0 0.0 0.005 0.7161 $\pm$ 0.0300
