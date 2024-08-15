# DIFFormer: Diffusion-based (Graph) Transformers

The official implementation for ICLR23 paper "DIFFormer: Scalable (Graph) Transformers Induced by Energy Constrained Diffusion".

Related material: [[Paper](https://arxiv.org/pdf/2301.09474.pdf)], [Blog [Chinese](https://zhuanlan.zhihu.com/p/622970740) | [English](https://medium.com/towards-data-science/how-to-build-graph-transformers-with-o-n-complexity-d507e103d30a)], [[Video](https://www.bilibili.com/video/BV1dP411C7Ti/?share_source=copy_web&vd_source=28f0a1823e05d5df3685cb9737bba371)] 

DIFFormer is a general-purpose encoder that can be used to compute instance representations with their latent/observed interactions accommodated.

This work is built upon [NodeFormer](https://github.com/qitianwu/NodeFormer) (NeurIPS22) which is a scalable Transformer for large graphs with linear complexity. 

## What's news

[2023.03.01] We release the early version of our codes for node classification.

[2023.03.09] We release codes for image/text classification and spatial-temporal prediction.

[2023.07.03] I gave a talk on LOG seminar about scalable graph Transformers. See the online video [here](https://www.bilibili.com/video/BV1dP411C7Ti/?share_source=copy_web&vd_source=28f0a1823e05d5df3685cb9737bba371).

[2024.08.15] We add codes for heterophily graphs (chameleon, squirrel and actor). See the training script in `./node classification/run.sh`. 

[2024.08.15] We extend the implementation for accommodating a batch of graphs as input, and add codes for particle physics datasets in `./physical particle`. See the [guideline](https://github.com/qitianwu/DIFFormer/tree/extension?tab=readme-ov-file#where-difformer-can-be-used).

## Model Overview

DIFFormer is motivated by an energy-constrained diffusion process which encodes a batch of instances to their structured representations. At each step, the model will first estimate pair-wise influence (i.e., attention) among arbitrary instance pairs (regardless of whether they connected by an input graph) and then update instance embeddings by feature propagation. The feed-forward process can be treated as a diffusion process that minimizes the global energy.

<img width="700" alt="image" src="https://user-images.githubusercontent.com/22075007/232401434-e433a273-2083-4ac8-ad82-e9e15dd51d49.png">

In specific, the DIFFormer's architecture is depicted by the following figure where one DIFFormer layer comprises of global attention, GCN convolution and residual link. The global attention is our key design including two instantiations: DIFFormer-s and DIFFormer-a.

<img width="700" alt="image" src="https://files.mdnice.com/user/23982/0f71e990-acbc-4706-aca3-680628f8ac92.png">

We implement the model in `difformer.py` where the DIFFormer-s (resp. DIFFormer-a) corresponds to `kernel = 'simple' (resp. 'sigmoid')`. The differences of two model versions lie in the global attention computation where DIFFormer-s only requires $O(N)$ complexity and DIFFormer-a requires $O(N^2)$, illustrated by the figure below where the red color marks the computation bottleneck. 

<img width="700" alt="image" src="https://files.mdnice.com/user/23982/3c433a8d-faf4-45f7-a4bd-c599e3288077.png">

## Where DIFFormer can be used?

We demonstrate the model on four different types of tasks: graph-based node classification, image and text classification, spatial-temporal prediction and particle property prediction. 
Beyond these scenarios, DIFFormer can be used as a general-purpose encoder for various applications including but not limited to:

- 1): **Encoding a graph**: given node features $X$ and graph adjacency $A$, the model outputs node embeddings $Z$ or predictions $\hat Y$. Please refer to codes in `./node classification` or `./spatial-temporal` for how to use it.

```python
      model = DIFFormer(in_channels, hidden_channels, out_channels, use_graph=True)
      z = model(x, edge_index) # x: [num_nodes, in_channels], edge_index: [2, E], z: [num_nodes, out_channels]
```

- 2): **Encoding a batch of instances (w/o graph structures, as a set)**: given a batch of instances $X$ (e.g., images), the model outputs instance embeddings $Z$ or predictions $\hat Y$. Please refer to codes in `./image and text` for how to use it.

```python
      model = DIFFormer(in_channels, hidden_channels, out_channels, use_graph=False)
      z = model(x, edge_index=None) # x: [num_inst, in_channels], z: [num_inst, out_channels]
```

- 3): **Encoding a batch of graphs (graphs can be disconnected with each other)**: given a batch of graphs, where their node features and graph adjacency are stacked as one big node feature matrix $\overline X$ and one big adjacency matrix $\overline A$ (diagonal block), 
the model outputs the embeddings $\overline Z$ for nodes in all graphs within the batch. Please refer to codes in `./physical particle` for how to use it.

```python
    model = DIFFormer_v2(in_channels, hidden_channels, out_channels, use_graph=True)
    z = model(x, edge_index, n_nodes) # x: [num_nodes, in_channels], edge_index: [2, E], n_nodes: [num_graphs], z: [num_nodes, out_channels]
```

- As plug-in encoder backbone for computing representations in latent space under a large framework for various downstream tasks (generation, prediction, decision, etc.).

## Dependence

Our implementation is based on Pytorch and Pytorch Geometric.
Please refer to `requirements.txt` in each folder for preparing the required packages.

## Datasets

We apply our model to three different tasks and consider different datasets.

- For ***node classification*** and ***image/text classification***, we provide an easy access to the used datasets in the [Google drive](https://drive.google.com/drive/folders/1sWIlpeT_TaZstNB5MWrXgLmh522kx4XV?usp=sharing) 
except two large graph datasets, OGBN-Proteins and Pokec, which can be automatically downloaded running the training/evaluation codes. 

*(for two image datasets CIFAR and STL, we use a self-supervised pretrained model (ResNet-18) to obtain the embeddings of images as input features)*

- For ***spatial-temporal prediction***, the datasets can be automatically downloaded from Pytorch Geometric Temporal.

- For ***particle property prediction***, the datasets can be automatically downloaded and processed when running our training pipeline.

Following [here](https://github.com/qitianwu/DIFFormer#how-to-run-our-codes) for how to get the datasets ready for running our codes.

## How to run our codes

1. Install the required package according to `requirements.txt` in each folder (notice that the required packages are different in each task)

2. Create a folder `../data` and download the datasets from [here](https://drive.google.com/drive/folders/1sWIlpeT_TaZstNB5MWrXgLmh522kx4XV?usp=sharing)
(For OGBN-Proteins, Pokec, spatial-temporal and particle datasets, the datasets will be automatically downloaded)

3. To train the model from scratch and evaluate on specific datasets, one can refer to the scripts `run.sh` in each folder.

4. To directly reproduce the results on two large datasets (the training can be time-consuming), we also provide the [checkpoints](https://drive.google.com/drive/folders/1sKIMSS9KrTsWazO_QLY7t84kcjrRNuxo?usp=sharing) of DIFFormer on OGBN-Proteins and Pokec.
One can download the trained models into `../model/` and run the scripts in `node classification/run_test_large_graph.sh` for reproducing the results. 

- For Pokec, to ensure obtaining the result as ours, one need to download the fixed splits from [here](https://drive.google.com/drive/folders/1in2___ubLLCo4f9resuM8qln6gsz7sS4?usp=sharing) to `../data/pokec/split_0.5_0.25`.


### Citation

If you find our codes useful, please consider citing our work

```bibtex
      @inproceedings{
        wu2023difformer,
        title={{DIFFormer: Scalable (Graph) Transformers Induced by Energy Constrained Diffusion},
        author={Qitian Wu and Chenxiao Yang and Wentao Zhao and Yixuan He and David Wipf and Junchi Yan},
        booktitle={International Conference on Learning Representations (ICLR)},
        year={2023}
        }
```
