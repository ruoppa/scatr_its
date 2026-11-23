[![arXiv](https://img.shields.io/badge/arXiv-2611.26715-b31b1b.svg)](https://doi.org/10.48550/arXiv.2611.26715)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

<div align="center">
<h1>SCATR-ITS: a scalable transformer for large-area individual tree segmentation from LiDAR point clouds</h1>

  **Lassi Ruoppa**<sup>1,2,*</sup>, **Verneri Seppänen**<sup>1</sup>, **Antero Kukko**<sup>1</sup>, **Juha Hyyppä**<sup>1</sup>

  <div><sup>1</sup><i>Department of Remote Sensing and Photogrammetry, Finnish Geospatial Research Institute FGI</i></div>
  <div><sup>2</sup><i>Department of Computer Science, Aalto University</i></div>

</div>
<br></br>

SCATR-ITS is a scalable transformer model for efficient individual tree segmentation from LiDAR point clouds covering large geographical areas. Based on the ForestFormer3D framework, SCATR-ITS incorporates several modifications designed to accelerate inference and reduce computational resource requirements while maintaining segmentation accuracy. The model achieves segmentation accuracy comparable to the state of the art set by ForestFormer3D on the FGI-EMIT and FOR-InstanceV2 datasets, while offering up to 8.5× faster inference and a substantially lower VRAM footprint during both training and inference. When applied to Finnish national laser scanning data (20 points/m<sup>2</sup>), SCATR-ITS achieves an average segmentation speed of 13.8 min/km<sup>2</sup>, approximately 18.6× faster than ForestFormer3D. Read the full preprint [here](https://doi.org/10.48550/arXiv.2611.26715).

<sup>*</sup>Corresponding author and maintainer of this repository.

## News

* **23.11.2026:** The SCATR-ITS codebase will be migrated from [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) to [Pointcept](https://github.com/pointcept/pointcept) to enable compatibility with more recent versions of PyTorch and CUDA, since MMDetection3D is no longer actively maintained. The Pointcept implementation will be merged into the main branch in the coming months, after which the MMDetection3D-based version will be deprecated. Star the repository to stay updated with the latest releases of SCATR-ITS.

## 1. Installation

The code has been tested on Ubuntu 22.04. Using Docker for installation is recommended. We provide a [Dockerfile](Dockerfile) that builds an image with all required libraries installed. Before building the image, ensure you have installed the following:
* [Docker](https://docs.docker.com/engine/install/) (see [here](https://docs.docker.com/desktop/setup/install/windows-install/) for Windows)
* [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (see [here](https://docs.docker.com/desktop/features/gpu/) for GPU support on Windows)

To build the Docker image, run:
```bash
docker build -t scatr_its .
```
Once the image has been built, start the container interactively using e.g. the following command:
```bash
docker run -it --rm \
    --gpus <gpus_to_use> \
    --shm-size=128g \
    --mount type=bind,source=$HOME/<path_to_data>,target=/workspaces/scatr_its/data/raw_data/<dataset_name> \
    --user root \
    scatr_its
```
The `<gpus_to_use>` argument specifies which GPUs are made available to the container (see the [Docker documentation](https://docs.docker.com/engine/containers/gpu/)). The `<path_to_data>` argument specifies the location of the point cloud data in the local filesystem, while `<dataset_name>` specifies the corresponding dataset directory inside the container. For further details on which data should be mounted and how to use the configuration files, see [Section 6.](#6-full-example). For convenience, we provide a [devcontainer.json](.devcontainer/devcontainer.json) configuration file to support [development within containers](https://code.visualstudio.com/docs/devcontainers/containers) using VS Code.

## 2. Data preprocessing

The original point cloud data should be mounted to the `data/raw_data` directory under the name of the dataset. The dataset name and the directories containing the different data splits must be set in the dataset configuration file under `dataset` and `dataset.preprocess`, respectively. The input file format must also be specified under `dataset.preprocess`. The LAS, LAZ, and PLY formats are all supported. **NOTE:** although PLY is supported, we recommend using LAS/LAZ since the IO is generally much faster and the format is overall more suitable for real world LiDAR point clouds. Some data splits can be empty, and files belonging to different splits can be stored in the same directory. The names of the point cloud files belonging to each split should be listed in the dataset configuration file under `dataset.data_split`. An example of how the point cloud data can be organized is shown below:
```
raw_data/
└── <dataset_name>/
    ├── <training_dir_name>
    │   ├── <data_name_1>.las
    │   ├── <data_name_2>.las
    │   └── ...
    ├── <validation_dir_name>
    │   ├── <data_name_3>.las
    │   └── ...
    └── <test_dir_name>
        ├── <data_name_4>.las
        └── ...
```

To preprocess the data for training the superpoint generator, run:
```bash
python -m preprocess.run --data_config=<path_to_dataset_config> --config=<path_to_superpoint_model_config> <additional_arguments>
```

To preprocess the data and generate superpoints for training/inference with SCATR-ITS, run:
```bash
python -m preprocess.run --data_config=<path_to_dataset_config> --config=<path_to_superpoint_generator_config> --checkpoint=<path_to_superpoint_generator_checkpoint> -sp <additional_arguments>
```

The preprocessing script supports the following additional arguments:
| **Argument** | **Description** |
| --- | --- |
| `-u`, `--unlabeled_mode` | Set if input data has no ground truth labels. |
| `-f`, `--force_rewrite`  | Force preprocessing of all files. Set if the underlying data or preprocessing settings have changed. By default the preprocessing skips files when corresponding preprocessed outputs exist. |

## 3. Inference

Following preprocessing, run the following command to perform inference using SCATR-ITS:
```bash
python test.py --data_config=<path_to_dataset_config> --config=<path_to_scatr_its_config> --checkpoint=<path_to_scatr_its_checkpoint> <additional_arguments>
```
Configure the desired output data format in the dataset configuration file under `dataset.output` and the accuracy computation settings under `dataset.accuracy`.

The inference script supports the following additional arguments:
| **Argument** | **Description** |
| --- | --- |
| `-u`, `--unlabeled_mode` | Set if data has no ground truth labels. This also skips all accuracy computations after inference. |
| `-c`, `--cast_to_original` | Cast the predicted labels back to original input point clouds by adding the prediction attributes to the original files. By default, the inference discards all point cloud attributes that are not used by the model. Note that casting predictions can be quite slow, particularly for large files. |
| `-d`, `--deterministic_alg` | Set `torch.use_deterministic_algorithms` to True for improved reproducibility. Note that this may slightly increase inference time, and all sources of randomness cannot be eliminated. Using this option is generally not recommended outside benchmarking. The accuracy metrics reported in the paper were obtained with `-d` set. |
| `--force_nn` | Use KDTree based nearest-neighbor interpolation when casting predictions to the original point clouds. If not set, an approximate method that may fail to find correspondences for some points is used instead. Enabling this option is recommended unless the point clouds are very large, in which case nearest-neighbor interpolation becomes prohibitively slow. Ignored unless `-c` is set. |
| `--prec` | Decimal precision used in approximate casting. Ignored when `--force_nn` is set. |

## 4. Training

Following preprocessing, run the following command to train a superpoint generator:
```bash
python train.py --data_config=<path_to_dataset_config> --config=<path_to_superpoint_generator_config>  --experiment_name=<experiment_name> -sp <additional_arguments>
```
Similarly, to train a SCATR-ITS model, run:
```bash
python train.py --data_config=<path_to_dataset_config> --config=<path_to_scatr_its_config>  --experiment_name=<experiment_name> <additional_arguments>
```
Training logs and checkpoints are saved under `data/experiments/<dataset_name>/<experiment_name>`.

The training script supports the following additional arguments:
| **Argument** | **Description** |
| --- | --- |
| `--resume` | Resume training from an existing checkpoint. Specify either the full path to a checkpoint or use `auto` to automatically select the latest checkpoint corresponding to <experiment_name>. |

## 5. Model zoo

Download links for pretrained SCATR-ITS and superpoint generator weights are provided for each of the three datasets used in the paper.

### 5.1 SCATR-ITS

| **Dataset** | **Model configuration** | **Dataset configuration** | **Checkpoint** |
| --- | --- | --- | --- |
| FGI-EMIT | [scatr_its.py](configs/models/scatr_its.py) | [FGI-EMIT.yaml](configs/datasets/FGI-EMIT.yaml) | [Download](https://drive.google.com/file/d/16IfGYLaXgorZ-R8yp3M6DtCA8wIEC6ne/view) |
| FOR-InstanceV2 (las) | [scatr_its.py](configs/models/scatr_its.py) | [FIV2_las.yaml](configs/datasets/FIV2_las.yaml) | [Download](https://drive.google.com/file/d/1LUM-W61I69oqs-Q6IXExqlc9_lEFuGAf/view) |
| ALS-XL-20-pts | [scatr_its.py](configs/models/scatr_its.py) | [ALS-XL-20-pts.yaml](configs/datasets/ALS-XL-20-pts.yaml) | [Download](https://drive.google.com/file/d/1gUS0pkRwDZNrdb7Dlb9Z75zr8JQ6Bdld/view) |

### 5.2 Superpoint generator

| **Dataset** | **Model configuration** | **Dataset configuration** | **Checkpoint** |
| --- | --- | --- | --- |
| FGI-EMIT | [superpoint_generator.py](configs/models/superpoint_generator.py) | [FGI-EMIT.yaml](configs/datasets/FGI-EMIT.yaml) | [Download](https://drive.google.com/file/d/19CMonIp8hXKR74wOZ_Z_khHT6KNkJU_I/view) |
| FOR-InstanceV2 (las) | [superpoint_generator.py](configs/models/superpoint_generator.py) | [FIV2_las.yaml](configs/datasets/FIV2_las.yaml) | [Download](https://drive.google.com/file/d/127zVL7vuWYy3NZcLMr8a_78M-Lf5oqBc/view) |
| ALS-XL-20-pts | [superpoint_generator.py](configs/models/superpoint_generator.py) | [ALS-XL-20-pts.yaml](configs/datasets/ALS-XL-20-pts.yaml) | [Download](https://drive.google.com/file/d/17iOASl4R_CKYIOmAjSzVOXK4j5aEsHrM/view) |


## 6. Full example

This section provides a step-by-step guide to segmenting unlabeled forest data with SCATR-ITS. This example uses weights pretrained on the FGI-EMIT dataset (suitable for high-density ALS data). To use different pretrained weights, simply replace the corresponding checkpoint and configuration files. As a starting point, we assume that you have successfully built the SCATR-ITS Docker image and downloaded the pretrained weights for both SCATR-ITS and the superpoint generator. For this example, we use an arbitrary dataset named "ForestData" but in practice the dataset can have any name.

**Step 1: Configuration**

We assume that the ForestData dataset is stored in a local directory named `data/ForestData/` and that all input files are in the LAS format. Since we only perform inference, the entirety of ForestData is considered to be in the test split. We use the dataset configuration file [template_FGI-EMIT.yaml](configs/datasets/template_FGI-EMIT.yaml), which serves as a template for segmenting data using a model pretrained on FGI-EMIT. Equivalent templates are also provided for the FOR-InstanceV2 and ALS-XL-20-pts datasets.

We modify the configuration template locally to match the structure of ForestData. Since in this case all point clouds are stored in the same directory, we set the training, validation, and test split directories to `null`:
```yaml
dataset:
  dataset_name: ForestData
  type: ForestDataset
  data_root: /workspaces/scatr_its/data

  preprocess:
    raw_data_dir: raw_data
    preprocessed_data_dir: preprocessed_data
    input_data_dir: input_data
    training_dir: ~
    validation_dir: ~
    test_dir: ~
    feat_to_load: ~
    max_n_points: ~
    file_extension: las
    ...
  ...
```
Next, we list the names of all files to be segmented under the test split, leaving the other splits empty:
```yaml
dataset:
  dataset_name: ForestData
  ...
  data_split:
    training: []
    validation: []
    test: [
      <data_1>,
      <data_2>,
      ...
    ]
```
Finally, we set `dataset.output.file_extension` to the desired output file format. In this example, we simply use the same format as the input, i.e. LAS.

Once the configuration file has been edited, we place it in the `data/configs/` directory. Similarly, we store the pretrained SCATR-ITS and superpoint generator chekpoints in a directory named `data/configs/`. The resulting (local) directory structure is as follows:
```
data/
├── ForestData/
│   ├── <data_1>.las
│   ├── <data_2>.las
│   └── ...
├── checkpoints/
│   ├── fgi_emit_8500.pth
│   ├── sp_fgi_emit_1000.pth
│   └── ...
└── configs/
    ├── template_FGI-EMIT.yaml
    └── ...
```
Note that this is only an example, and in practice the structure can be almost anything, provided that the corresponding directories are mounted to the container appropriately.

**Step 2: Starting the container**

Run the following command to start the container:
```bash
docker run -it --rm \
    --gpus all \
    --shm-size=128g \
    --mount type=bind,source=$HOME/data/ForestData,target=/workspaces/scatr_its/data/raw_data/ForestData \
    --mount type=bind,source=$HOME/data/checkpoints,target=/workspaces/scatr_its/data/checkpoints \
    --mount type=bind,source=$HOME/data/configs,target=/workspaces/scatr_its/configs/datasets \
    --user root \
    scatr_its
```

**Step 3: Preprocessing**

Next, we generate superpoints and preprocess the data for segmentation by running the following command:
```bash
python -m preprocess.run --data_config=configs/datasets/template_FGI-EMIT.yaml --config=configs/models/superpoint_generator.py --checkpoint=data/checkpoints/sp_fgi_emit_1000.pth -sp -u
```

**Step 4: Inference**

Once preprocessing is complete, we perform segmentation with SCATR-ITS by running the following command:
```bash
python test.py --data_config=configs/datasets/template_FGI-EMIT.yaml --config=configs/models/scatr_its.py --checkpoint=data/checkpoints/fgi_emit_8500.pth -u
```

**Step 5: Results and notes**

Once segmentation is complete, the results can be found within the container in `data/predictions/FGI-EMIT/ForestDataset/`. Copy the results to a local directory, and make sure you remove the container once you no longer need it. If you want the final results to contain all attributes from the original point clouds, add the `-c` flag to the inference command (see [Section 3.](#3-inference)).

If your dataset includes ground truth labels and you want to evaluate the segmentation accuracy, remove the `-u` flag from both the preprocessing and inference commands. In addition, ensure that the values of `dataset.preprocess.instance_field` and `dataset.preprocess.semantic_field` in the dataset configuration file are set to the names of the ground truth instance and semantic label fields, respectively.

## 7. Miscellaneous

The repository contains several miscellaneous utility scripts, which are described in further detail below.

### 7.1 Constructing FIV2_las

To convert the FOR-InstanceV2 dataset from its original PLY format to the LAS format used in the paper and specified in [FIV2_las.yaml](configs/datasets/FIV2_las.yaml), run:
```bash
python util/migrate_fiv2_to_las.py --src_path=<path_to_original_data> --dest_path=<path_for_las_data>
```
The script also divides the BlueCat region training point cloud into smaller subsections and computes crown categories for all instances.

### 7.2 Casting after inference

To cast predictions back to the original point clouds after inference, run:
```bash
python util/cast_predictions --data_config=<path_to_dataset_config> --config=<path_to_scatr_its_config>
```
Equivalent to the `-c` option during inference, the script supports the `-u`, `--force_nn`, and `--prec` options.

### 7.3 Computing accuracy after inference

To compute the accuracy of generated superpoints for a given dataset, run:
```bash
python accuracy.py --data_config=<path_to_dataset_config> --config=<path_to_superpoint_generator_config> -sp
```

Similarly, to compute the accuracy of instance segmentation results from the latest run for the given dataset, run:
```bash
python accuracy.py --data_config=<path_to_dataset_config> --config=<path_to_scatr_its_config>
```

## 8. Known issues

* **Segmented point clouds become corrupted when the ouput file format is set to LAZ and the files are sufficiently large:** This is caused by an [issue in the laspy version used by the codebase](https://github.com/laspy/laspy/issues/352). As a workaround, the code automatically switches the file format to LAS when files exceed a certain point count (see [pc_io.py](util/pc_io.py)). If you encounter this issue, try reducing the value of `_MAX_LAZ_SIZE` or setting the output format to LAS. The laspy version will be updated in the upcoming migration of SCATR-ITS to Pointcept, which is expected to resolve the issue.
* **Processing point clouds in the PLY format is slower than LAS/LAZ:** IO is handled with the [plyfile](https://github.com/dranjan/python-plyfile) python library, which seems to be somewhat slow for large files. We recommend using LAS/LAZ at all times, since the format is overall more suitable for processing real world LiDAR point clouds.

## 9. Acknowledgements

SCATR-ITS contains original and/or modified source code from the following repositories:
* Significant parts of the SCATR-ITS model code are based on the official implementations of [ForestFormer3D](https://github.com/SmartForest-no/ForestFormer3D) and [OneFormer3D](https://github.com/filaPro/oneformer3d).
* The GPU-accelerated algorithm for finding connected components in graphs, along with several graph utility functions, are from the [Superpoint Transformer repository](https://github.com/drprojects/superpoint_transformer)
* The code for computing superpoint boundary precision (BP) and boundary recall (BP) is adopted from the official [Superpoint Graph repository](https://github.com/loicland/superpoint_graph).

## 10. Citation

If you use the code in this repository for your research, please cite the associated paper:
```bibtex
@article{ruoppa2026benchmarking,
    title = {{SCATR-ITS: a scalable transformer for large-area individual tree segmentation from LiDAR point clouds}},
    author = {Ruoppa, Lassi and Seppänen, Verneri and Kukko, Antero and Hyyppä, Juha},
    journal = {arXiv preprint arXiv:2611.26715},
    year = {2026},
    doi = {10.48550/arXiv.2611.26715},
}
```

If you use the pretrained weights, also consider citing the the relevant papers listed below.

**FGI-EMIT:**
```bibtex
@article{ruoppa2026benchmarking,
    title = {{Benchmarking individual tree segmentation using multispectral airborne laser scanning data: The FGI-EMIT dataset}},
    author = {Ruoppa, Lassi and Hietala, Tarmo and Seppänen, Verneri and Taher, Josef and Hakala, Teemu and Yu, Xiaowei and Kukko, Antero and Kaartinen, Harri and Hyyppä, Juha},
    journal = {ISPRS J. Photogramm. Remote Sens.},
    volume = {236},
    pages = {569--605},
    year = {2026},
    doi = {10.1016/j.isprsjprs.2026.04.021},
}
```

**FOR-InstanceV2:**
```bibtex
@inproceedings{xiang2025forestformer3d,
    author = {Xiang, Binbin and Wielgosz, Maciej and Puliti, Stefano and Král, Kamil and Krůček, Martin and Missarov, Azim and Astrup, Rasmus},
    title = {{ForestFormer3D: A Unified Framework for End-to-End Segmentation of Forest LiDAR 3D Point Clouds}},
    booktitle = {2025 IEEE/CVF International Conference on Computer Vision (ICCV)},
    year={2025},
    pages={24717--24727},
    doi={10.1109/ICCV51701.2025.02292}
}
```

**ALS-XL-20-pts:**

In addition to the FGI-EMIT and FOR-InstanceV2 datasets, ALS-XL-20-pts contains data from the EvoMS and AerialTrees datasets:
```bibtex
@article{ruoppa2025unsupervised,
    author = {Ruoppa, Lassi and Oinonen, Oona and Taher, Josef and Lehtomäki, Matti and Takhtkeshha, Narges and Kukko, Antero and Kaartinen, Harri and Hyyppä, Juha},
    title = {{Unsupervised deep learning for semantic segmentation of multispectral LiDAR forest point clouds}},
    journal = {ISPRS J. Photogramm. Remote Sens.},
    year = {2025},
    volume = {228},
    pages = {694--722},
    doi = {10.1016/j.isprsjprs.2025.07.038},
}


@article{li2025itsnet,
    author = {Li, Bowen and Pang, Yong and Kükenbrink, Daniel and Wang, Luo and Kong, Dan and Marty, Mauro},
    title = {{ITS-Net: A platform and sensor agnostic 3D deep learning model for individual tree segmentation using aerial LiDAR data}},
    journal = {ISPRS J. Photogramm. Remote Sens.},
    year = {2026},
    volume = {231},
    pages = {719--744},
    doi = {10.1016/j.isprsjprs.2025.11.019},
}
```

## 11. License

This repository contains code distributed under multiple licenses:
* All code derived from the original ForestFormer3D source code is distributed under the [CC BY-NC 4.0 license](LICENSE-CC-BY-NC-4.0)
* Everything else is distributed under the [MIT license](LICENSE)