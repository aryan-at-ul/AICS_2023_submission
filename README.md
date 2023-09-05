# Connecting the Dots: Graph Neural Network Powered Ensembly and Classification of Medical Images

## Abstract
Deep learning models have demonstrated remarkable results for various computer vision tasks, including the realm of medical imaging. However, their application in the medical domain is limited due to the requirement for large amounts of training data, which can be both challenging and expensive to obtain. To mitigate this, pre-trained models have been fine-tuned on domain-specific data, but such an approach can suffer from inductive biases. Furthermore, deep learning models struggle to learn the relationship between spatially distant features and their importance, as convolution operations treat all pixels equally. Pioneering a novel solution to this challenge, we employ the Image Foresting Transform to optimally segment images into superpixels. These superpixels are subsequently transformed into graph-structured data, enabling the proficient extraction of features and modeling of relationships using Graph Neural Networks (GNNs). Our method harnesses an ensemble of three distinct GNN architectures to boost its robustness. In our evaluations targeting pneumonia classification, our methodology surpassed prevailing Deep Neural Networks (DNNs) in performance, all while drastically cutting down on the parameter count. This not only trims down the expenses tied to data but also accelerates training and minimizes bias. Consequently, our proposition offers a sturdy, economically viable, and scalable strategy for medical image classification, significantly diminishing dependency on extensive training data sets.

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

