# E-SegNet: Advanced E-Structured Networks for Accurate 2D and 3D Medical Image Segmentation
This repository contains official implementation for the paper titled "**E-SegNet: Advanced E-Structured Networks for Accurate 2D and 3D Medical Image Segmentation**".
In this work, we have built two models for 2D and 3D medical image segmentation, respectively. The 2D E-SegNet is validated on the Synapse (multi-organ segmentation), ACDC (cardiac segmentation), and Kvasir-Seg (polyp segmentation) datasets, while the 3D E-SegNet is validated on the Synapse (3D) and NIH Pancreas (pancreatic segmentation) datasets.
## Environment
1. Create a new conda environment with python version 3.8.18:
   ```
   conda create -n "e_segnet" python=3.8.18
   conda activate e_seg_net
   ```

2. Install PyTorch and torchvision
   ```
   We recommend an evironment with pytorch==2.0.0+cu118, torchvision==0.15.1+cu118, torchaudio==2.0.1+cu118
   ```

3. Other packages can be found in the requirements.txt file, or you can directly install them using:
   ```
   pip install -r requirements.txt
   ```
   
## 2D Medical Image Segmentation
### Dataset Preparation
1. Download the Synapse and ACDC dataset from the link [Synapse and ACDC dataset](https://github.com/Beckschen/TransUNet).
2. Change the directory to the 2D folder (assuming you are currently in the E-SegNet main directory):
    ```
   cd 2D/
   ```

### Training and Testing
3. Run the code below to train or test 2D E-SegNet on the **Synapse** dataset.
   ```
   python train_synapse.py --root_path [Your Synapse dataset path]
   python test_synapse.py --volume_path [Your Synapse dataset path] --load_checkpoint [Path to the trained weights]
   ```
4. Run the code below to train or test 2D E-SegNet on the **ACDC** dataset.
   ```
   python train_acdc.py --root_path [Your ACDC dataset path]
   python test_acdc.py --volume_path [Your ACDC dataset path] --load_checkpoint [Path to the trained weights]
   ```
5. Run the code below to train or test 2D E-SegNet on the **Kvasir-Seg** dataset.
   ```
   python train_kvasir.py
   ```

## 3D Medical Image Segmentation
### Dataset Preparation
1. Download the Synapse and Pancreas dataset from the link [Synapse 3D dataset](https://github.com/Amshaker/unetr_plus_plus), [NIH Pancreas dataset](https://github.com/xmindflow/deformableLKA).
You should strictly follow the given steps.

### Pre-trained Weights Download
2. Download the [Swin-T](https://github.com/SwinTransformer/Video-Swin-Transformer?tab=readme-ov-file) pre-trained weights.
3. Change the directory to the 3D folder (assuming you are currently in the E-SegNet main directory).
   ```
   cd 3D/
   ```

### Training and Testing on the **Synapse 3D** dataset
4. Place the downloaded pre-trained weights into ./Synapse/unetr_pp/training/network_training/pre_checkpoint
5. Run the code below to train or test 3D E-SegNet on the **Synapse 3D** dataset.
   ```
   cd Synapse/training_scripts/
   bash run_training_synapse.sh
   ```
   or Inference.
   ```
   cd ../evaluation_scripts/
   bash run_evaluation_synapse.sh
   ```
   
### Training and Testing on the **NIH Pancreas** dataset
6. Place the downloaded pre-trained weights into NII_pancreas/pre_checkpoint
7. Run the code below to train or test 3D E-SegNet on the **NIH Pancreas** dataset.
    ```
   python train_pancreas.py --root_path [Your NIH Pancreas dataset path]
   python test_pancreas.py --root_path [Your NIH Pancreas dataset path] --load_checkpoint [Path to the trained weights]
   ```

This repository references [AgileFormer](https://github.com/sotiraslab/AgileFormer), [D-LKA Net](https://github.com/xmindflow/deformableLKA), [unetr_plus_plus](https://github.com/Amshaker/unetr_plus_plus?tab=readme-ov-file). Thanks to them.