# DIFFormer: Diffusion-based (Graph) Transformers

The official implementation for ICLR23 paper "DIFFormer: Scalable (Graph) Transformers Induced by Energy Constrained Diffusion"

## What's news

[2023.03.01] We release the early version of our codes for node classification.

[2023.03.09] We release codes for image/text classification and spatial-temporal prediction.

```bibtex
      @inproceedings{
        wu2023difformer,
        title={{DIFF}ormer: Scalable (Graph) Transformers Induced by Energy Constrained Diffusion},
        author={Qitian Wu and Chenxiao Yang and Wentao Zhao and Yixuan He and David Wipf and Junchi Yan},
        booktitle={International Conference on Learning Representations (ICLR)},
        year={2023}
        }
```

## Dependence

Our implementation is based on Pytorch and Pytorch Geometric.
Please refer to `requirements.txt` in each folder for preparing the required packages.

## Datasets

We apply our model to three different tasks and consider different datasets.

- For ***node classification*** and ***image/text classification***, we provide an easy access to the used datasets in the [Google drive](https://drive.google.com/drive/folders/1sWIlpeT_TaZstNB5MWrXgLmh522kx4XV?usp=sharing) 
except two large graph datasets, OGBN-Proteins and Pokec, which can be automatically downloaded running the training/evaluation codes.

- For ***spatial-temporal prediction***, the datasets can be automatically downloaded from Pytorch Geometric Temporal.

Following [here](https://github.com/qitianwu/DIFFormer#how-to-run-our-codes) for how to get the datasets ready for running our codes.

## How to run our codes

1. Install the required package according to `requirements.txt` in each folder (notice that the required packages are different in each task)

2. Create a folder `../data` and download the datasets from [here](https://drive.google.com/drive/folders/1sWIlpeT_TaZstNB5MWrXgLmh522kx4XV?usp=sharing)
(For OGBN-Proteins, Pokec and three spatial-temporal datasets, the datasets will be automatically downloaded)

3. To train the model from scratch and evaluate on specific datasets, one can refer to the scripts `run.sh` in each folder.

4. To directly reproduce the results on two large datasets (the training can be time-consuming), we also provide the [checkpoints](https://drive.google.com/drive/folders/1sKIMSS9KrTsWazO_QLY7t84kcjrRNuxo?usp=sharing) of DIFFormer on OGBN-Proteins and Pokec.
One can download the trained models into `../model/` and run the scripts in `node classification/run_test_large_graph.sh` for reproducing the results. 

- For Pokec, to ensure obtaining the result as ours, one need to download the fixed splits from [here](https://drive.google.com/drive/folders/1in2___ubLLCo4f9resuM8qln6gsz7sS4?usp=sharing) to `../data/pokec/split_0.5_0.25`.


### Citation

If you find our codes useful, please consider citing our work

```bibtex
      @inproceedings{
        wu2023difformer,
        title={{DIFF}ormer: Scalable (Graph) Transformers Induced by Energy Constrained Diffusion},
        author={Qitian Wu and Chenxiao Yang and Wentao Zhao and Yixuan He and David Wipf and Junchi Yan},
        booktitle={International Conference on Learning Representations (ICLR)},
        year={2023}
        }
```

