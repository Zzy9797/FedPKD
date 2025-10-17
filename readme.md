# FedPKD
Personalized federated learning with personalized knowledge distillation based on local empirical risk

## Code Descriptions
- run.sh: Quick strat
- train.py: Main code of the algorithm
  
## Experiments
Our experiments are implemented with open-source PyTorch 1.13.1, on an NVIDIA GeForce RTX 4090 platform. 

### Results of Dirichlet Distribution: 

Dirichlet distribution is a typical data splitting principle in FL, which effectively mimics the heterogeneity of data in real applications. We conduct the experiments under Dirichlet distribution data partition to compare the performance among the state-of-the-arts methods.

#### simplecnn as backbone

Table 1: Accuracy (%) comparisons of image classification tasks on the datasets CIFAR-10, CIFAR-100, SVHN, and Fashion-MNIST datasets using simplecnn as the client local model under the Dirichlet data partitioning.

 <span style="white-space:nowrap;">Method&emsp;&emsp;&emsp;</span> |<span style="white-space:nowrap;">CIFAR-10&emsp;&emsp;&emsp;</span>  |<span style="white-space:nowrap;">CIFAR-100&emsp;&emsp;&emsp;</span>  |<span style="white-space:nowrap;">SVHN&emsp;&emsp;&emsp;</span> |<span style="white-space:nowrap;">Fashion-MNIST&emsp;&emsp;&emsp;</span>
|---------------|------------------|------------------|------------------|------------------|
| FedAvg        | 62.92            | 27.78            | 85.86            | 84.51            |
| FedAvg-FT     | 84.09            | 50.58            | 90.11            | 96.33            |
| FedProx       | 62.25            | 27.87            | 85.98            | 82.79            |
| FedProx-FT    | 83.71            | 50.99            | 89.26            | 96.27            |
| FedAS         | 50.20            | 12.97            | 80.75            | 55.48            |
| CFL           | 83.84            | 49.12            | 89.53            | 96.64            |
| Per-FedAvg    | 84.02            | 50.38            | 89.82            | 96.30            |
| pFedMe        | 75.24            | 34.37            | 82.59            | 93.34            |
| FedAMP        | 75.49            | 31.04            | 67.62            | 93.13            |
| Ditto         | 83.78            | 50.33            | 90.13            | 95.97            |
| FedRep        | 83.47            | 50.15            | 89.32            | 96.44            |
| pFedHN        | 82.57            | 49.08            | 78.44            | 95.87            |
| FedRoD        | 83.49            | 47.96            | 89.13            | 96.46            |
| kNN-Per       | 70.05            | 25.84            | 86.21            | 91.87            |
| pFedGraph     | 84.28            | 51.63            | 89.59            | 96.46            |
| FedPKD        | **86.10**±1.90   | **54.08**±0.46   | **90.60**±0.66   | **96.84**±0.08   |


#### ResNet18 as backbone

Table 2: Accuracy (%) comparisons of image classification tasks on the datasets CIFAR-10, CIFAR-100, SVHN, and Fashion-MNIST datasets using ResNet18 as the client local model under the Dirichlet data partitioning.
 <span style="white-space:nowrap;">Method&emsp;&emsp;&emsp;</span> |<span style="white-space:nowrap;">CIFAR-10&emsp;&emsp;&emsp;</span>  |<span style="white-space:nowrap;">CIFAR-100&emsp;&emsp;&emsp;</span>  |<span style="white-space:nowrap;">SVHN&emsp;&emsp;&emsp;</span> |<span style="white-space:nowrap;">Fashion-MNIST&emsp;&emsp;&emsp;</span>
|---------------|------------------|------------------|------------------|------------------|
| FedAvg        | 86.96            | 63.47            | 94.36            | 89.67            |
| FedAvg-FT     | 93.72            | 80.88            | 95.34            | 97.66            |
| FedProx       | 85.68            | 63.37            | 94.64            | 88.44            |
| FedProx-FT    | 93.91            | 80.95            | 95.54            | 97.79            |
| CFL           | 92.94            | 80.27            | 94.42            | 97.49            |
| Per-FedAvg    | 59.82            | 10.75            | 42.72            | 67.99            |
| pFedMe        | 89.40            | 71.78            | 90.61            | 96.30            |
| FedAMP        | 85.54            | 53.10            | 90.16            | 96.23            |
| Ditto         | 93.94            | 77.79            | 93.47            | 97.79            |
| FedRep        | 93.60            | 78.66            | 95.27            | 97.62            |
| FedRoD        | 83.61            | 48.79            | 95.06            | 97.59            |
| kNN-Per       | 71.55            | 29.46            | 94.71            | 95.54            |
| pFedGraph     | 94.19            | 79.72            | 93.70            | 97.53            |
| FedALA        | 93.54            | 80.22            | 95.45            | 97.61            |
| FedPKD        | **94.60**±1.40   | **81.02**±0.22   | **95.89**±0.34   | **97.94**±0.06   |

### Results of Homogeneous Data Partition:
When data is homogeneous, excessive personalization can sometimes affect model performance. To validate the effectiveness of proposed FedPKD under data homogeneity, we conduct the experiments on CIFAR-10, CIFAR-100, SVHN and Fashion-MNIST under independent and identically distributed data partition.

#### simplecnn as backbone

Table 3: Accuracy (%) comparisons of image classification task on the datasets CIFAR-10, CIFAR-100, SVHN, and Fashion-MNIST datasets using simplecnn as the client local model under homogeneous data partitioning.
 <span style="white-space:nowrap;">Method&emsp;&emsp;&emsp;</span> |<span style="white-space:nowrap;">CIFAR-10&emsp;&emsp;&emsp;</span>  |<span style="white-space:nowrap;">CIFAR-100&emsp;&emsp;&emsp;</span>  |<span style="white-space:nowrap;">SVHN&emsp;&emsp;&emsp;</span> |<span style="white-space:nowrap;">Fashion-MNIST&emsp;&emsp;&emsp;</span>
|---------------|------------------|------------------|------------------|------------------|
| FedAvg        | 67.12            | 31.10            | 88.69            | 87.27            |
| FedAvg-FT     | 63.09            | 25.47            | 86.13            | 86.23            |
| FedProx       | 67.07            | 30.55            | 88.56            | 87.21            |
| FedProx-FT    | 61.93            | 25.15            | 86.05            | 85.96            |
| CFL           | 60.55            | 19.31            | 84.45            | 86.38            |
| Per-FedAvg    | 63.24            | 25.18            | 86.08            | 86.28            |
| pFedMe        | 47.48            | 13.18            | 63.96            | 77.32            |
| FedAMP        | 45.49            | 10.07            | 66.48            | 74.42            |
| Ditto         | 65.35            | 29.41            | 88.26            | 86.85            |
| FedRep        | 62.88            | 21.53            | 86.29            | 85.73            |
| FedRoD        | 62.07            | 18.71            | 85.72            | 85.50            |
| kNN-Per       | 67.01            | 31.04            | 88.88            | 87.37            |
| pFedGraph     | 67.37            | 31.16            | 88.53            | 87.25            |
| FedPKD        | **67.39**±0.78   | **31.84**±0.71   | **88.92**±0.13   | **87.39**±0.17   |

#### ResNet18 as backbone

Table 4: Accuracy (%) comparisons of image classification tasks on the datasets CIFAR-10, CIFAR-100, SVHN, and Fashion-MNIST datasets using ResNet18 as the client local model under homogeneous data partitioning.
 <span style="white-space:nowrap;">Method&emsp;&emsp;&emsp;</span> |<span style="white-space:nowrap;">CIFAR-10&emsp;&emsp;&emsp;</span>  |<span style="white-space:nowrap;">CIFAR-100&emsp;&emsp;&emsp;</span>  |<span style="white-space:nowrap;">SVHN&emsp;&emsp;&emsp;</span> |<span style="white-space:nowrap;">Fashion-MNIST&emsp;&emsp;&emsp;</span>
|---------------|------------------|------------------|------------------|------------------|
| FedAvg-FT     | 86.31            | 62.98            | 93.94            | 92.44            |
| FedProx-FT    | 86.21            | 63.88            | 93.95            | 92.49            |
| CFL           | 82.15            | 63.76            | 91.76            | 91.95            |
| pFedMe        | 77.97            | 48.21            | 88.64            | 88.02            |
| FedAMP        | 67.60            | 29.93            | 87.82            | 87.26            |
| Ditto         | 90.42            | 33.62            | 95.50            | 93.66            |
| FedRep        | 86.29            | 54.29            | 93.86            | 92.17            |
| FedRoD        | 85.97            | 52.16            | 93.77            | 92.19            |
| kNN-Per       | 67.44            | 31.86            | 95.92            | 93.97            |
| pFedGraph     | 75.42            | 33.77            | 90.73            | 92.50            |
| FedALA        | 90.41            | 64.81            | 95.41            | 93.93            |
| FedPKD        | **90.85**±0.27   | **66.31**±0.50   | **95.93**±0.11   | **93.89**±0.08   |

## Acknowledgments
Our work is based on the following work, thanks for the code:

https://github.com/MediaBrain-SJTU/pFedGraph