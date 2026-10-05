# C-MAPSS-VLM-Prognostics
VLM-assisted RUL prediction framework combining numerical models, Qwen2.5-VL visual analysis, and deterministic fusion on NASA C-MAPSS.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Md-Rafsan-Isdani/C-MAPSS-VLM-Prognostics/blob/main/vlm_prognostics.ipynb)

Code for a remaining useful life (RUL) prediction framework on the NASA C-MAPSS turbofan dataset. The idea is to combine standard ML/DL RUL models with a vision-language model (VLM) that looks at sensor trend plots the way an engineer would, and then check whether that visual evidence actually improves the prediction.

## How it works

The pipeline has two branches that meet at a fusion step.

1. **Numerical models.** Eight models are trained on the preprocessed sensor data: Linear Regression, Random Forest, XGBoost, BiLSTM, TCN, MS-TCN, MS-TCN+BiLSTM and a Transformer. Preprocessing includes dropping near-constant sensors, regime-wise normalization for FD002/FD004, RUL capping at 125 and 30-cycle sliding windows.
2. **Visual branch.** For each test engine, the raw values of all 21 sensors are plotted and passed to Qwen2.5-VL-7B-Instruct. The VLM returns a structured JSON describing the degradation trend, abnormal sensors, where the terminal phase seems to start, and a qualitative RUL band (high / medium / low / critical).
3. **Fusion.** The final RUL is calculated in Python, not by a language model. Each model is weighted by the inverse of its validation RMSE, outliers are removed with a MAD rule, and the VLM's RUL band can shift the weights by at most 15%.
4. **Evidence analysis.** Qwen2.5-7B-Instruct reads the predictions, the VLM output and the fused result, and gives a short assessment with an accept / caution / reject recommendation. It has no way to change the number.

## Files

| File | Description |
|---|---|
| `vlm_prognostics.ipynb` | Main Colab notebook, runs FD001 to FD004 end to end |
| `vlm_prognostics.py` | Same code as a plain Python script |

## Requirements

You need a CUDA GPU. I used Google Colab with an L4; an A100 works too. Both language models run in 4-bit, so they fit on one GPU together.

Main packages: `torch`, `transformers>=4.49`, `accelerate`, `bitsandbytes`, `qwen-vl-utils`, `scikit-learn`, `xgboost`, `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`.

The first cell of the notebook installs whatever is missing.

## Dataset

Download the C-MAPSS dataset from the [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/). The zip should contain `train_FD00X.txt`, `test_FD00X.txt` and `RUL_FD00X.txt` for all four subsets.

| Subset | Test engines | Operating conditions | Fault modes |
|---|---|---|---|
| FD001 | 100 | 1 | 1 |
| FD002 | 259 | 6 | 1 |
| FD003 | 100 | 1 | 2 |
| FD004 | 248 | 6 | 2 |

## Running it

1. Open the notebook in Colab and switch to a GPU runtime.
2. Run all cells and upload `CMAPSSData.zip` when asked. To load it from Google Drive instead, set `UPLOAD_ZIP = False` and edit the path.
3. Results are saved to `/content/artifacts/`.

A full run over all four subsets takes a few hours, since both 7B models run once per test engine. If Colab disconnects, run the notebook again. Any subset that already finished is loaded from disk and skipped.

To try things out quickly first, set `MAX_TEST_ENGINES_PER_SUBSET = 15`. These test runs are saved in a separate folder so they don't get mixed up with real results.

Settings you might want to change:

| Parameter | Default | What it does |
|---|---|---|
| `SUBSETS` | FD001 to FD004 | Which subsets to run |
| `RUL_CAP` | 125 | RUL cap for training and evaluation |
| `WINDOW_SIZE` | 30 | Window length in cycles |
| `VLM_ADJUSTMENT_BOUND` | 0.15 | Maximum weight change the VLM can cause |
| `MAX_TEST_ENGINES_PER_SUBSET` | None | Limit engines for a quick test |
| `SEED` | 191 | Random seed |

## Outputs

Each subset gets its own folder under `/content/artifacts/<SUBSET>/` with:

- `metrics_table.csv`: RMSE, MAE, NASA score and Pearson r for every method
- `ablation_table.csv`: how much each step of the fusion adds
- `significance_tests.csv`: Wilcoxon and paired t-tests against the best single model
- `fusion_results_full.csv`: per-engine predictions, weights and LLM comments
- `vlm_outputs.json` and `vlm_raw_text.json`: the VLM responses
- `run_summary.json`: everything above in one file

The cross-subset comparison is saved in `/content/artifacts/summary/`.

## Results

| Method | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| Best single model | | | | |
| Constrained fusion (numeric only) | | | | |
| Constrained fusion + VLM | | | | |

RMSE in cycles. Will be updated after the final runs.

## Notes and limitations

- C-MAPSS has no label for when degradation actually starts. The VLM's terminal phase estimate is compared against the first cycle where RUL drops to 30 or below, which is a practical reference rather than ground truth.
- The confidence score is a fixed weighted combination of model agreement and VLM output. It is not a calibrated probability.
- Seeds are fixed and decoding is greedy, so runs repeat on the same hardware. Different GPUs or library versions can still shift the last few decimals.
- The LLM explanations have not been checked against expert judgment yet.

## Citation

If you find this useful, please cite:

```bibtex
@misc{rafsan2026vlmrul,
  author = {Rafsan, Md},
  title  = {VLM-Assisted Prognostic Decision Framework for Remaining Useful Life Prediction},
  year   = {2026},
  url    = {https://github.com/Md-Rafsan-Isdani/C-MAPSS-VLM-Prognostics}
}
```

## Acknowledgments

Thanks to the NASA Ames Prognostics Center of Excellence for the C-MAPSS dataset, and to the Qwen team for the open models.

## License

MIT
