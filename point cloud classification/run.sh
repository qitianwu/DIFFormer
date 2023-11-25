dropout_list=(0.3 0.4 0.5)
decay_list=(5e-5 1e-4 5e-4 1e-3)


for decay in ${decay_list[@]};do
for dropout in ${dropout_list[@]};do
python main.py --dataset tau3mu --batch_size 128 --lr 0.01 \
    --method difformer --num_layers 2 --hidden_channels 64 --dropout $dropout --weight_decay $decay \
    --graph_pooling mean --kernel simple2 --use_bn --use_residual --use_graph --use_weight \
    --device 1 --runs 3 --epochs 100
done
done



python main.py --dataset tau3mu --batch_size 2048 --lr 0.01 \
    --method difformer --num_layers 2 --hidden_channels 64 --dropout 0.4 --weight_decay 5e-5 \
    --graph_pooling mean --kernel simple --use_bn --use_residual --use_graph --use_weight \
    --device 2 --runs 2 --epochs 100

dropout_list=(0.3 0.4 0.5)
decay_list=(5e-5 1e-4 5e-4 1e-3)


for decay in ${decay_list[@]};do
for dropout in ${dropout_list[@]};do
python main.py --dataset tau3mu --batch_size 128 --lr 0.01 \
    --method difformer --num_layers 2 --hidden_channels 64 --dropout $dropout --weight_decay $decay \
    --graph_pooling mean --kernel sigmoid --use_bn --use_residual --use_graph --use_weight \
    --device 1 --runs 3 --epochs 100
done
done

# dropout_list=(0.3 0.4 0.5)
lr_list=(0.1 0.05 0.5)
decay_list=(5e-5 1e-4 5e-4 1e-3)

for decay in ${decay_list[@]};do
for lr in ${lr_list[@]};do
python main.py --dataset tau3mu --batch_size 128 --lr $lr \
    --method difformer --num_layers 2 --hidden_channels 64 --dropout 0.5 --weight_decay $decay \
    --graph_pooling mean --kernel sigmoid --use_bn --use_residual --use_graph --use_weight \
    --device 0 --runs 2 --epochs 40 --display_step 10
done
done

python main.py --dataset tau3mu --batch_size 128 --lr 0.01 \
    --method difformer --num_layers 2 --hidden_channels 64 --dropout 0.5 --weight_decay 5e-4 \
    --graph_pooling mean --kernel sigmoid --use_bn --use_residual --use_graph --use_weight \
    --device 2 --runs 2 --epochs 101

dropout_list=(0.3 0.4 0.5)
decay_list=(1e-4 5e-4 1e-3 5e-3)
lr_list=(0.01 0.005)

for dropout in ${dropout_list[@]};do
for decay in ${decay_list[@]};do
for lr in ${lr_list[@]};do
python main.py --dataset synmol --batch_size 128 --lr $lr \
    --method difformer --num_layers 2 --hidden_channels 64 --dropout $dropout --weight_decay $decay \
    --graph_pooling mean --kernel simple2 --use_bn --use_residual --use_graph --use_weight \
    --device 0 --runs 3 --epochs 150
done
done
done

python main.py --dataset synmol --batch_size 2048 --lr 0.005 \
    --method difformer --num_layers 2 --hidden_channels 64 --dropout 0.4 --weight_decay 1e-4 \
    --graph_pooling mean --kernel simple --use_bn --use_residual --use_graph --use_weight \
    --device 1 --runs 2 --epochs 101

dropout_list=(0.3 0.4 0.5)
decay_list=(1e-4 5e-4 1e-3 5e-3)
lr_list=(0.01 0.005)

for dropout in ${dropout_list[@]};do
for decay in ${decay_list[@]};do
for lr in ${lr_list[@]};do
python main.py --dataset synmol --batch_size 128 --lr $lr \
    --method difformer --num_layers 2 --hidden_channels 64 --dropout $dropout --weight_decay $decay \
    --graph_pooling mean --kernel sigmoid --use_bn --use_residual --use_graph --use_weight \
    --device 0 --runs 3 --epochs 150
done
done
done

python main.py --dataset synmol --batch_size 128 --lr 0.005 \
    --method difformer --num_layers 2 --hidden_channels 64 --dropout 0.3 --weight_decay 5e-4 \
    --graph_pooling mean --kernel sigmoid --use_bn --use_residual --use_graph --use_weight \
    --device 1 --runs 2 --epochs 101

python main.py --dataset actstrack --batch_size 128 --lr 0.01 \
    --method difformer --num_layers 2 --hidden_channels 64 --dropout 0.5 --weight_decay 5e-4 \
    --graph_pooling mean --kernel simple --use_bn --use_residual --use_graph --use_weight \
    --device 1 --runs 2 --epochs 101

python main.py --dataset actstrack --batch_size 128 --lr 0.01 \
    --method difformer --num_layers 2 --hidden_channels 64 --dropout 0.5 --weight_decay 5e-4 \
    --graph_pooling mean --kernel sigmoid --use_bn --use_residual --use_graph --use_weight \
    --device 1 --runs 2 --epochs 101


python main.py --dataset tau3mu --batch_size 128 \
    --method gcn --num_layers 3 --hidden_channels 64 --dropout 0.5 --weight_decay 5e-4 \
    --graph_pooling mean \
    --device 0 --runs 5

python main.py --dataset synmol --batch_size 128 \
    --method gcn --num_layers 3 --hidden_channels 64 --dropout 0.5 --weight_decay 5e-4 \
    --graph_pooling mean \
    --device 0 --runs 5

python main.py --dataset actstrack --batch_size 128 \
    --method gcn --num_layers 2 --hidden_channels 64 --dropout 0.5 --weight_decay 5e-4 \
    --graph_pooling mean \
    --device 1 --runs 5

python main.py --dataset plbind --batch_size 128 \
    --method gcn --num_layers 2 --hidden_channels 64 --dropout 0.5 --weight_decay 5e-4 \
    --graph_pooling mean \
    --device 2 --runs 5

dropout_list=(0.3 0.4 0.5)
decay_list=(1e-4 5e-4 1e-3)
lr_list=(0.01 0.005)

for dropout in ${dropout_list[@]};do
for decay in ${decay_list[@]};do
for lr in ${lr_list[@]};do
python main.py --dataset plbind --batch_size 128 --lr $lr \
    --method difformer --num_layers 2 --hidden_channels 64 --dropout $dropout --weight_decay $decay \
    --graph_pooling mean --kernel simple2 --use_bn --use_residual --use_graph --use_weight \
    --device 2 --runs 3 --epochs 150
done
done
done

python main.py --dataset plbind --batch_size 128 --lr 0.01 \
    --method difformer --num_layers 2 --hidden_channels 64 --dropout 0.5 --weight_decay 5e-4 \
    --graph_pooling mean --kernel simple2 --use_bn --use_residual --use_graph --use_weight \
    --device 2 --runs 3 --epochs 150

dropout_list=(0.3 0.4 0.5)
decay_list=(1e-4 5e-4 1e-3)
lr_list=(0.01)

for dropout in ${dropout_list[@]};do
for decay in ${decay_list[@]};do
for lr in ${lr_list[@]};do
python main.py --dataset plbind --batch_size 128 --lr $lr \
    --method difformer --num_layers 2 --hidden_channels 64 --dropout $dropout --weight_decay $decay \
    --graph_pooling mean --kernel sigmoid --use_bn --use_residual --use_graph --use_weight \
    --device 1 --runs 3 --epochs 150
done
done
done

python main.py --dataset plbind --batch_size 128 --lr 0.01 \
    --method difformer --num_layers 2 --hidden_channels 64 --dropout 0.5 --weight_decay 5e-4 \
    --graph_pooling mean --kernel sigmoid --use_bn --use_residual --use_graph --use_weight \
    --device 1 --runs 3 --epochs 150