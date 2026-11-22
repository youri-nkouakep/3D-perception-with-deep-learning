# 3D Perception with Deep Learning

Teaching a machine to understand the world in 3D: recognizing objects from point clouds, estimating depth from a single image, and detecting objects in a LiDAR scene.

The project goes from a single object to a full street scene in three steps:

| Step | Task | Data | Status |
|---|---|---|---|
| 1 | Point cloud classification (PointNet) | ModelNet40 | Done, **88.6%** |
| 2 | Monocular depth estimation | NYU Depth v2 | In progress |
| 3 | 3D object detection in LiDAR scans | KITTI | Planned |

## Step 1: PointNet on ModelNet40

![ModelNet40 point cloud samples](docs/modelnet40_samples.png)

ModelNet40 contains 12,311 3D models across 40 classes (airplane, chair, table, guitar...). Each mesh is turned into a cloud of 1,024 points sampled uniformly on its surface, then centered and scaled into the unit sphere.

PointNet ([Qi et al., CVPR 2017](https://arxiv.org/abs/1612.00593)) reads these points directly, without turning them into images or voxels. The same small network processes every point on its own, then a max-pooling summarizes the whole set: the result does not depend on the order of the points, which is checked in the tests.

### Results

| | Overall accuracy | Class average |
|---|---|---|
| This project (25 epochs, CPU, no 64x64 T-Net) | **88.6%** | 83.1% |
| Original paper (250 epochs, GPU, full model) | 89.2% | 86.2% |

The model gets within 0.6 points of the paper with ten times fewer epochs, trained in 5 hours on an Intel i3 laptop CPU with no GPU.

![Learning curves](docs/pointnet_curves.png)

The jumps at epochs 11 and 21 match the points where the learning rate is halved. Test accuracy stays above training accuracy because training uses augmented data (scaling, shifting, noise) and dropout, while the test runs on the raw clouds.

### Where the model gets it wrong

<img src="docs/pointnet_per_class.png" width="560" alt="Accuracy per class">

The errors are not random. They focus on objects that have almost the same shape:

| True class | Predicted as | Share of samples |
|---|---|---|
| flower_pot | plant | 55% |
| cup | vase | 30% |
| radio | glass_box | 25% |
| wardrobe | bookshelf | 25% |
| night_stand | dresser | 22% |

A flower pot almost always holds a plant, and a night stand looks a lot like a smaller dresser. Since every cloud is scaled to the same size, the real size of the object is lost, and that is exactly what tells these classes apart. The weakest classes also have only 20 test samples, so a single mistake costs 5 points.

The full confusion matrix is in [docs/pointnet_confusion.png](docs/pointnet_confusion.png) and the per-class numbers in [docs/pointnet_results.json](docs/pointnet_results.json).

### Reproduce

```bash
conda env create -f environment.yml
conda activate perception
pip install open3d   # optional, for the interactive 3D view

# ModelNet40.zip (1.9 GB): http://modelnet.cs.princeton.edu, unzip into data/raw/
python scripts/prepare_modelnet40.py
python scripts/train_pointnet.py --epochs 25 --lr-step 10 --no-feature-transform
python scripts/evaluate_pointnet.py
```

An interrupted run continues with `--resume`. On a GPU, the full model (without `--no-feature-transform`) over 250 epochs follows the paper's setup.

## Project structure

```
src/perception/
  datasets/modelnet.py   .off reader, surface sampling, PyTorch Dataset
  models/pointnet.py     PointNet, T-Net and regularization
scripts/                 data preparation, training, evaluation, visualization
tests/                   unit tests (pytest)
docs/                    figures and results
```

## Tests

```bash
pytest
```

Among other things, the tests check that the prediction does not change when the points are shuffled, and that the model can memorize a small batch.
