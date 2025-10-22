# CMA

## Concept Matching with Agent for Out-of-Distribution Detection

Yuxiao Lee, Xiaofeng Cao, Jingcai Guo, Wei Ye, Qing Guo, Yi Chang

Concept Matching with Agents (CMA) introduces neutral textual “agents” into the zero-shot OOD detection pipeline to reshape the CLIP softmax distribution. Instead of relying solely on the maximum similarity between an image and in-distribution (ID) labels, CMA jointly normalises over ID labels and agent prompts, allowing out-of-distribution (OOD) images to be “claimed” by neutral concepts and therefore separated from ID data without supervision. 

## [Project Page](https://github.com/yuxiaoLeeMarks/CMA) | [Paper (https://ojs.aaai.org/index.php/AAAI/article/view/32481/34636)]()

![overview](images/CMA_overview.png)

## Highlights

- **Agent-augmented scoring**: CMA computes the posterior of the most compatible ID label while neutral agent prompts expand the softmax denominator, suppressing OOD confidence.
- **Zero-shot and training-free**: builds entirely on pre-trained CLIP encoders; no gradients or fine-tuning required.
- **Flexible prompt design**: customise the number and wording of agent prompts through CLI flags or text files.
- **Full benchmarking suite**: retain support for the large-scale ImageNet benchmarks and classical baselines shipped with the original project.

## Repository Map

- `eval_ood_detection.py`: unified entry point for evaluating CMA and other scores (`CMA` is the default).
- `clip_based_OOD_detection.py`: wrapper that invokes CMA scoring directly for backwards-compatible scripts.
- `utils/`: data loaders, scoring utilities, plotting helpers, and common routines reused across experiments.
- `scripts/`: shell helpers (`eval_cma.sh`, `eval_mcm.sh`) for standard experiment presets.
- `paper_outline.md`: technical summary of the CMA method and experimental setup.

## Environment Setup

- Ubuntu 20.04 / Python 3.8 (other recent CPython versions should also work)
- PyTorch ≥ 1.10 with CUDA support if available
- Hugging Face `transformers`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `pandas`, `tqdm`

Install dependencies with your preferred package manager, for example:

```bash
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118  # adjust CUDA tag if needed
pip install transformers scipy scikit-learn matplotlib seaborn pandas tqdm
```

CLIP checkpoints are loaded automatically from Hugging Face (`openai/clip-vit-base-patch16`, etc.). ViT checkpoints for auxiliary experiments can be referenced through the same hub.

## Data Preparation

The default dataset root is `./datasets`. ImageNet-based benchmarks follow the structure introduced in the original project.

### In-distribution datasets

- ImageNet-1k, ImageNet-10, ImageNet-20, ImageNet-100
- CUB-200, Stanford Cars, Food-101, Oxford-IIIT Pet

Use `create_imagenet_subset.py` to construct the ImageNet subsets:

```bash
python create_imagenet_subset.py --in_dataset ImageNet10 --src-dir datasets/ImageNet --dst-dir datasets
python create_imagenet_subset.py --in_dataset ImageNet20 --src-dir datasets/ImageNet --dst-dir datasets
python create_imagenet_subset.py --in_dataset ImageNet100 --src-dir datasets/ImageNet --dst-dir datasets
```

### Out-of-distribution datasets

- iNaturalist, SUN, Places, and DTD (from the large-scale OOD benchmark of Huang et al., 2021)

Place these datasets under `datasets/ImageNet_OOD_dataset/` with the expected sub-directory names (`iNaturalist`, `SUN`, `Places`, `dtd/images`, ...).

## Running CMA

Launch the standard CMA evaluation on ImageNet-1k with:

```bash
sh scripts/eval_cma.sh eval_cma ImageNet
```

or call the Python entry directly:

```bash
python eval_ood_detection.py --in_dataset ImageNet --name eval_cma --gpu 0
```

Key CMA options:

- `--prompt-template`: format string used to convert ID class names into textual prompts (default `a photo of a {}`).
- `--agent-ratio` / `--num-agents`: control the quantity of agent prompts relative to the number of ID classes (`k = 1` by default).
- `--agent-prompts`: pipe-separated custom prompts supplied inline (`"a photo of a thing.|a photo of a scene we live in."`).
- `--agent-prompts-file`: newline-delimited file containing agent prompts for reproducible sweeps.

Changes can also be driven through `clip_based_OOD_detection.py`, which forwards all arguments while enforcing `--score CMA`.

## Baselines and Ablations

- **MCM baseline**: preserve the original Maximum Concept Matching score via `--score MCM` or `sh scripts/eval_mcm.sh <exp> <dataset> MCM`.
- **Other scores**: energy, max-logit, entropy, variance, and Mahalanobis scores remain available for comparison.

Results are written to `results/<ID>/<score>/<model>_.../<run_name>/`. Figures and logs follow the same layout as the baseline project.

## Conceptual Summary

1. Encode images with the CLIP image encoder and normalise features.
2. Tokenise and encode ID labels plus neutral agent prompts through the CLIP text encoder.
3. Compute cosine similarities, apply temperature scaling, and form a softmax over ID+agent concepts.
4. CMA takes the maximum probability over ID labels as the confidence score; large agent activations suppress ID confidence for OOD samples.
5. Evaluate metrics such as AUROC and FPR95 using the provided utilities.

The recommended configuration mirrors the paper: `k = 1` (one agent per ID class), neutral prompts drawn from the default set, and zero-shot inference on CLIP ViT-B/16 features.

## Citation

If you find CMA useful, please cite the accompanying paper. 

```
@inproceedings{lee2025concept,
  title={Concept Matching with Agent for Out-of-Distribution Detection},
  author={Lee, Yuxiao and Cao, Xiaofeng and Guo, Jingcai and Ye, Wei and Guo, Qing and Chang, Yi},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={5},
  pages={4562--4570},
  year={2025}
}
```

---

Maintained by the CMA authors. Contributions and issues are welcome via pull requests.

Concurrently, the framework for this project originated from [MCM](https://github.com/deeplearning-wisc/MCM/tree/main), and their open-source contributions are gratefully acknowledged.


