# quadtreecnn
Scalable fire segmentation for high resolution aerial images using CNNs and Quad-tree search

The input of CNNs are fixed-size and images are often downsized to feed into the network to avoid computatinal complexity. This causes the loss of information and eliminating small fire regions which are important to detect.

This code uses a quad-tree search algorithm to segment different scales of fire in high resolution aerial images. It is able to detect small fire areas by prograssively zooming the incident region while keeping a fixed size for the segmentation network


