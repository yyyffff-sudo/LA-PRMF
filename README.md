# Layer-Adaptive Gating and Prompt-Enhanced Proxy Generation for Robust Multimodal Sentiment Analysis with Missing Modalities

PyTorch implementation of the paper:
**Layer-Adaptive Gating and Prompt-Enhanced Proxy Generation for Robust Multimodal Sentiment Analysis with Missing Modalities (LA-PRMF).**

This work is built upon P-RMF and further optimizes its proxy-driven dynamic injection and proxy modality generation modules. This is a reorganized version of the code; if you find any bugs, please open an issue or contact me. Thanks.

## Content
- [Project Structure](#project-structure)
- [Data Preparation](#data-preparation)
- [Environment](#environment)
- [Training](#training)
- [Evaluation](#evaluation)
- [Ablation Study](#ablation-study)
- [Note](#note)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)

## Project Structure
```
LA-PRMF/
├── configs/
│   ├── train_mosi.yaml
│   ├── train_mosei.yaml
│   ├── train_sims.yaml
│   ├── eval_mosi.yaml
│   ├── eval_mosei.yaml
│   └── eval_sims.yaml
├── core/
│   ├── dataset.py
│   ├── losses.py
│   ├── metric.py
│   ├── optimizer.py
│   ├── scheduler.py
│   └── utils.py
├── models/
│   ├── P_RMF.py
│   ├── bert.py
│   ├── basic_layers.py
│   └── generate_proxy_modality.py
├── train.py
├── evaluation.py
├── requirements.txt
└── README.md
```

## Data Preparation
**MOSI / MOSEI / CH-SIMS download:** please refer to [MMSA](https://github.com/thuiar/MMSA) for the unaligned feature/label files. CMU-MOSI and CMU-MOSEI are also available through the [CMU Multimodal SDK](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK).

This code uses the unaligned pickle files (`unaligned_50.pkl` for MOSI/MOSEI and `unaligned_39.pkl` for CH-SIMS). The datasets are **not** included in this repository.

You also need a local copy of the pretrained BERT weights: `bert-base-uncased` for MOSI/MOSEI and `bert-base-chinese` for CH-SIMS (e.g. downloaded from Hugging Face).

**Before running, edit the corresponding config file** in `configs/` and update these two fields to your own paths:

```yaml
dataset:
  dataPath: /your/path/to/unaligned_50.pkl              # dataset feature file
model:
  feature_extractor:
    bert_pretrained: /your/path/to/bert-base-uncased    # local BERT weights
```

## Environment
The reported results use **Python 3.11.7** and **PyTorch 2.5.1 (CUDA 12.4)** on a single **NVIDIA RTX 4060** GPU.

```bash
# 1. create and activate an environment
conda create -n laprmf python=3.11.7 -y
conda activate laprmf

# 2. install PyTorch 2.5.1 built for CUDA 12.4
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# 3. install the remaining dependencies
pip install -r requirements.txt
```

## Training
Run training by passing the corresponding config file to `train.py`:

```bash
# MOSI
python train.py --config_file configs/train_mosi.yaml

# MOSEI
python train.py --config_file configs/train_mosei.yaml

# CH-SIMS
python train.py --config_file configs/train_sims.yaml
```

To reproduce results over the three random seeds, override the seed from the command line:

```bash
python train.py --config_file configs/train_mosi.yaml
```

During training, 50% of the samples are randomly selected in each epoch and 0–100% of the information in each modality is randomly erased to simulate missing modalities. The model trains for 100 epochs with a batch size of 32. After each epoch the model is evaluated on the test set, and the best checkpoint for each metric is automatically saved to:

```
ckpt/<datasetName>/best_test_<key_eval>_<seed>.pth
```

for example `ckpt/mosi/best_test_Has0_acc_2_1111.pth`.

## Evaluation
In the testing phase the missing rate is fixed (set by `missing_rate_eval_test` in the eval config) to simulate practical scenarios such as long-term sensor failures.

Use `evaluation.py` with an eval config and a saved checkpoint. For example, to evaluate the binary-classification accuracy on MOSI:

```bash
python evaluation.py --config_file configs/eval_mosi.yaml \
                     --ckpt ckpt/mosi/best_test_Has0_acc_2_1111.pth \
                     --key_eval Has0_acc_2
```

To evaluate robustness across different missing rates, change `missing_rate_eval_test` in the eval config (e.g. 0.0, 0.1, …, 0.9) and rerun the command above.

Available `key_eval` values:
- MOSI / MOSEI: `Has0_acc_2`, `Non0_acc_2`, `Has0_F1_score`, `Non0_F1_score`, `Mult_acc_5`, `Mult_acc_7`, `MAE`, `Corr`
- CH-SIMS: `Mult_acc_2`, `Mult_acc_3`, `Mult_acc_5`, `F1_score`, `MAE`, `Corr`

## Ablation Study
The two proposed components and the original P-RMF behaviours are controlled by an optional `ablation` block in the config file. If the block is omitted, the full LA-PRMF model is used by default:

```yaml
ablation:
  use_grl: false             # remove the Gradient Reversal Layer (default: false)
  use_prompt: true           # modality-specific prompts in the VAEs (default: true)
  use_lawg: true             # Layer-Adaptive Weight Gating (default: true)
  independent_layers: true   # independent parameters per injection layer (default: true)
```

To reproduce the ablation rows in the paper, add the block to your training config and toggle the switches:

| Setting             | `use_lawg` | `use_prompt` |
|---------------------|:----------:|:------------:|
| LA-PRMF (full)      | true       | true         |
| w/o Prompt          | true       | false        |
| w/o LAWG            | false      | true         |
| w/o LAWG & Prompt   | false      | false        |

## Note
Regression metrics (e.g. MAE and Corr) and classification metrics (e.g. Acc-2 and F1) focus on different aspects of model performance: the checkpoint with the lowest sentiment-intensity error does not necessarily give the best classification accuracy. To comprehensively reflect each model's capability, the reported classification and regression metrics may correspond to the best-performing checkpoints of different epochs within the same training run. If you wish to compare all metrics at the same epoch, please rerun the code and record the metrics at a fixed epoch.

## Acknowledgement
This work is built upon P-RMF. We also thank the authors of [MMSA](https://github.com/thuiar/MMSA) and the providers of the CMU-MOSI, CMU-MOSEI, and CH-SIMS datasets.

## Citation
Please cite our paper if you find our work useful for your research:

```bibtex
@article{yang_laprmf,
  title   = {Layer-Adaptive Gating and Prompt-Enhanced Proxy Generation for Robust Multimodal Sentiment Analysis with Missing Modalities},
  author  = {Yang, Fan and Shen, Junfeng},
  journal = {Pattern Analysis and Applications},
  note    = {Under review},
  year    = {2026}
}
```
<!-- TODO: update the BibTeX (volume, pages, doi, year) once the paper is accepted/published -->
