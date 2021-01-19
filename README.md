# Sparse LiDAR Depth Completion
Code for `DAN-Conv: Depth Aware Non-local Convolution for LiDAR Depth Completion`.

## Requirments
```
pytorch >= 1.6.0
tqdm
fire
```

## Prepare training data
You need to download the entire [raw KITTI dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php) to train.

**Warning:** it weighs about **175GB**, so make sure you have enough space to unzip too!

Our default settings expect that you have converted the png images to jpeg.

The path containing KITTI datasets should be organized like this:
```
/path/to/kitti
    raw/
        2011_09_26/
            2011_09_26_drive_0001_sync/
                image_02/
                image_03/
    test_depth_completion_anonymous/
        image/
        velodyne_raw/
    val_selection_cropped/
        image/
        velodyne_raw/
        groundtruth_depth/
```

## Build kNN lib
```
cd nearest_neighbors
make
```

## Train on Single GPU
Run
```
python train.py --data_path /path/to/kitti --split full
```

## Train on multiple GPUs
Modify `nproc_per_node` in script `train_multi_gpus.sh` to the number of GPUs you have and adjust the batch size.

Run
```
. train_multi_gpus.sh
```
