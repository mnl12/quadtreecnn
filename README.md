# quadtreecnn
Scalable fire segmentation for high resolution aerial images using CNNs and Quad-tree search

The input of CNNs are fixed-size and images are often downsized to feed into the network to avoid computatinal complexity. This causes the loss of information and eliminating small fire regions which are important to detect.

This code uses a quad-tree search algorithm to segment different scales of fire in high resolution aerial images. It is able to detect small fire areas by prograssively zooming the incident region while keeping a fixed size for the segmentation network

Below, you can see an example. The white borders just indicate how the algorithm processes the image

![Sample image](https://github.com/mnl12/quadtreecnn/blob/main/Sample_images/6.jpg?raw=true)
![Predicted mask](https://github.com/mnl12/quadtreecnn/blob/main/Results/quadtree/quad_pred_mask6.png?raw=true)

# Installation:
To install using conda:

```
conda create -n fire_qtree python=3.8
conda activate fire_qtree
pip install --upgrade pip
pip install -r requirements.txt
```
