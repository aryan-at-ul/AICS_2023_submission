# From Pixels to Connections: Powering Pneumonia \\ Detection with Lean and Accurate GNNs in Medical Imaging

## Abstract
Deep learning models have demonstrated remarkable results for various computer vision tasks, including the realm of medical imaging. However, their application in the medical domain is limited due to the requirement for large amounts of training data, which can be both challenging and expensive to obtain. To mitigate this, pre-trained models have been fine-tuned on domain-specific data, but such an approach can suffer from inductive biases. Furthermore, deep learning models struggle to learn the relationship between spatially distant features and their importance, as convolution operations treat all pixels equally. To address this limitation, we propose a novel method that involves grouping local features into superpixels and converting them into graph-structured data. This enables efficient feature capture and relationship modelling using a Graph Neural Network (GNN). Our approach has been evaluated for efficient pneumonia classification and has been found to outperform state-of-the-art DNNs while considerably reducing parameter count. This, in turn, leads to decreased data-related expenses, faster training, and less bias. Thus, our proposed method presents a robust, cost-effective, and scalable solution for medical image classification, with reduced reliance on large amounts of training data.

### Create virtual enviornment 
```
conda env create -n graphenv -f environment.yml
conda activate graphenv
```


### To run an experiment 
```
python main.py  --cnn_model_name {resnet18/efficienet-b0/densenet121} --gnn_model {GCN/GAT/GIN}  --use_saved_state yes   --superpixel_number {5/10/50/100/150/300}  --train {yes/no/True/False}
```
Example:
In this example we are testing on graph created on image with 5 superpixels, features extracted from resnet18 and GNN model used is GCN

```
python main.py  --cnn_model_name resnet18  --gnn_model GCN  --use_saved_state yes   --superpixel_number 5  --train no
```

To replicate the results in colab use **replicate_experiments.ipynb**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1E2iWu7IsS2eK8jyZS1dD5ZK2cBsc6fly?usp=sharing)

To replicate ensemlby result use **ensembling_result.ipynb**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1aktjvqLi908s3VcKJVRDYa77ENQuo7AS?usp=sharing)

### To create new graph dataset from image, dowload chest_xray images and run the following commands
```
sh download.sh 

python main.py  --cnn_model_name {cnn_model_name}  --use_saved_state no  --superpixel_number {superpixel_value}
```
Example:
This will create a folder named chest_xray_graph and will create graph from images, using features of resnet18 and superpixel value of 5 i.e 5 node graph. 
```
python main.py  --cnn_model_name resnet18  --gnn_model GCN  --use_saved_state no  --superpixel_number 5
```

