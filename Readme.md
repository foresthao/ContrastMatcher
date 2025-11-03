## ContrastMatcher (CM)

ContrastMatcher: Adaptive Contrastive Provenance Graph Matching for Host-Based Attack Detection

ContrastMatcher is a project for graph contrastive learning and matching on DARPA TC scenarios. It learns graph-level representations via graph augmentations and self-supervised learning, then performs binary classification (match vs. non-match) with SVM/LR in downstream evaluation. Metrics include Accuracy and AUROC. The project covers data parsing and compression, contrastive training, and graph matching evaluation end-to-end.

### Code Structure

- **aug**: Contrastive training and evaluation

  - `aug.py`: Graph augmentations (node/edge drop, feature masking, k-core/k-truss, random walk/subgraph sampling, etc.)

  - `train.py`: Training loop and augmentation branch construction (two augmented views for contrastive learning)

  - `evaluate.py`: Extract graph-level embeddings with the encoder, run 10-fold CV using SVM/LR, report Accuracy/AUROC

  - `methods/GraphCL.py`, `methods/JOAO.py`: Method entrypoints (GraphCL/JOAO and OGSN/Attention variants)

  - `__init__.py`: Unified `Method` dispatcher, dataset loading, logging

- **experiment**: Experiment entrypoints and scripts

  - `cm_main.py`: Main entry; parses arguments, sets random seeds, creates `Method`, and starts training

- **opr**: Utilities for log compression/parsing and graph construction

- **data**: Data directory (e.g., DARPA TC E3)

- **ValGraphMatchDataset**: Packaging logic for evaluation datasets in graph matching

- `cm_env2.yml`, `cm_environment.yml`: Environment dependencies (conda)

- `log`: Training logs and intermediate outputs

### Workflow Overview

1) Entry: `experiment/cm_main.py`

   - Parse arguments (method, data paths, augmentation type, epochs, batch size, layers, sampling ratio, etc.)

   - Construct `aug.Method` to load datasets and initialize logging

2) Training (GraphCL/JOAO family)

   - `aug.train.train` selects augmentation via `--core`:

     - `ndrop`: edge drop; `edrop`: node drop; `fmask`: feature mask; `subsample`: BFS subgraph sampling

     - `kcore`/`ktruss`: structural subgraph augmentation (`KCoreSubgraph`/`KTrussSubgraph`)

     - `no`: disable augmentation

   - The encoder (`GConv`/`GConv_OGSN`/`GIN_OGSN_Attention`) extracts graph-level embeddings for two views; optimized with InfoNCE

3) Evaluation: `aug.evaluate.test_save_model_cm`

   - Package pairs `(graph_t, graph_g)` with labels via `ValGraphMatchDataset.GraphMatchDataset`

   - Concatenate the two graph embeddings as features; run 10-fold cross-validation (default SVM; optional standardization)

   - Log and aggregate Accuracy and AUROC (val/test)

### Quick Start

1) Environment (conda recommended)

```bash
conda env create -f cm_env2.yml

conda activate cm
```

2) Prepare Data
- Place DARPA TC data under the default paths, or specify the scenario via `--dataset_darpa_name`. 
3) Example Run

```bash
python experiment/cm_main.py \

  --method GraphCL_OGSN_ATT \

  --dataset_darpa_name trace_data \

  --feature origin_darpa \

  --ego 5 \

  --core ndrop \

  --times 5 \

  --epoch 50 \

  --batch_size 64 \

  --num_layer 2 \

  --pn 0.2 \

  --eval_model True
```

### Key Arguments (cm_main.py)

- **--method**: Training method. Supports `GraphCL`, `JOAO` and their `*_OGSN`, `*_OGSN_ATT` variants

- **--dataset_darpa_name**: Scenario name; automatically mapped to absolute data paths

- **--feature**: Feature scheme (currently `origin_darpa`)

- **--ego**: Neighborhood radius (used in preprocessing/sampling)

- **--core**: Augmentation type:

  - `ndrop` edge drop, `edrop` node drop, `fmask` feature mask, `subsample` subgraph sampling

  - `kcore`/`ktruss` structural subgraph augmentation, `no` to disable augmentation

- **--pn**: Retention ratio for nodes/edges/features (semantics vary by augmentation)

- **--epoch / --batch_size / --num_layer / --hid_units**: Training configuration and model size

- **--interval**: Evaluation interval (epochs)

- **--eval_model**: Whether to periodically evaluate during training (SVM 10-fold)

- **--save_model / --save_embed**: Save model weights or embeddings

### Logs and Outputs

- Training logs: `log/<method>_<dataset>_<ego>hop_<core>_<batch>_layer_<num>_pn_<pn>.log`

- Optional artifacts: model weights (pkl), embeddings (pkl)

- ROC curves can be generated via `test_SVM1`, exported to `roc_curve.pdf` by default

### Development Tips

- Augmentations live in `aug/train.py` and `aug/aug.py`. To add a custom augmentation, inherit from `Augmentor` and plug it into the `--core` branch.

- Encoders reside in `aug/model.py` (not shown here). `GConv_OGSN`/`GIN_OGSN_Attention` correspond to OGSN and attention variants, respectively.

- The evaluation entry `test_save_model_cm` concatenates embeddings and uses SVM by default; you can swap in a custom classifier.

### Citation

If this project is useful to your research, please acknowledge ContrastMatcher (CM). Please contact me at yanhaoforest@gmail.com.

The code will be publicly available all source codes after the paper is published. Stay tuned for updates!!!