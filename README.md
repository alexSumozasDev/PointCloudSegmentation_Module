# PCA PointCloud

A Python package for segmenting 3D point clouds using Principal Component Analysis (PCA). This package allows you to segment 3D point clouds into distinct regions based on the axes derived from PCA. You can use it to divide a point cloud into octants, slice-based regions, or segments based on eigenvalue direction (min/max).

## Features

- **PCA-based segmentation**: Segment a 3D point cloud into regions based on the principal axes computed using PCA.
- **Multiple segmentation methods**: 
  - **All PCA Segments**: Segments the point cloud into 8 octants based on the PCA axes.
  - **Min/Max Eigenvalue Segmentation**: Segments based on the eigenvectors of the PCA with the smallest/largest eigenvalue.
  - **Slice Segmentation**: Divides the point cloud into regions along the PCA axes, based on a specified number of slices.
  
## Installation

You can install the package directly from PyPI using pip:

```bash
pip install pca-pointcloud
