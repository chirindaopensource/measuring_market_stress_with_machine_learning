# **`README.md`**

# Algorithmic Monitoring: Measuring Market Stress with Machine Learning

<!-- PROJECT SHIELDS -->
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2602.07066-b31b1b.svg)](https://arxiv.org/abs/2602.07066)
[![Journal](https://img.shields.io/badge/Journal-ArXiv%20Preprint-003366)](https://arxiv.org/abs/2602.07066)
[![Year](https://img.shields.io/badge/Year-2026-purple)](https://github.com/chirindaopensource/measuring_market_stress_with_machine_learning)
[![Discipline](https://img.shields.io/badge/Discipline-Financial%20Econometrics%20%7C%20Machine%20Learning-00529B)](https://github.com/chirindaopensource/measuring_market_stress_with_machine_learning)
[![Data Sources](https://img.shields.io/badge/Data-CRSP%20%7C%20WRDS-lightgrey)](https://wrds-www.wharton.upenn.edu/)
[![Core Method](https://img.shields.io/badge/Method-Lasso--Logit%20%7C%20Expanding%20Window-orange)](https://github.com/chirindaopensource/measuring_market_stress_with_machine_learning)
[![Analysis](https://img.shields.io/badge/Analysis-Probabilistic%20Forecasting-red)](https://github.com/chirindaopensource/measuring_market_stress_with_machine_learning)
[![Validation](https://img.shields.io/badge/Validation-Brier%20Score%20%7C%20ECE%20%7C%20AUC-green)](https://github.com/chirindaopensource/measuring_market_stress_with_machine_learning)
[![Robustness](https://img.shields.io/badge/Robustness-Block%20Bootstrap%20%7C%20Nonlinear%20Benchmarks-yellow)](https://github.com/chirindaopensource/measuring_market_stress_with_machine_learning)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type Checking: mypy](https://img.shields.io/badge/type%20checking-mypy-blue)](http://mypy-lang.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=flat&logo=scipy&logoColor=white)](https://scipy.org/)
[![YAML](https://img.shields.io/badge/YAML-%23CB171E.svg?style=flat&logo=yaml&logoColor=white)](https://yaml.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-%23F37626.svg?style=flat&logo=Jupyter&logoColor=white)](https://jupyter.org/)
[![Open Source](https://img.shields.io/badge/Open%20Source-%E2%9D%A4-brightgreen)](https://github.com/chirindaopensource/measuring_market_stress_with_machine_learning)

**Repository:** `https://github.com/chirindaopensource/measuring_market_stress_with_machine_learning`

**Owner:** 2026 Craig Chirinda (Open Source Projects)

This repository contains an **independent**, professional-grade Python implementation of the research methodology from the 2026 paper entitled **"Algorithmic Monitoring: Measuring Market Stress with Machine Learning"** by:

*   **Marc Schmitt** (University of Oxford)

The project provides a complete, end-to-end computational framework for replicating the paper's findings. It delivers a modular, auditable, and extensible pipeline that executes the entire research workflow: from the ingestion and rigorous validation of CRSP micro-data to the construction of the Market Stress Probability Index (MSPI) using L1-regularized logistic regression, culminating in comprehensive out-of-sample evaluation and structural economic analysis.

## Table of Contents

- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Features](#features)
- [Methodology Implemented](#methodology-implemented)
- [Core Components (Notebook Structure)](#core-components-notebook-structure)
- [Key Callable: `execute_complete_mspi_research_pipeline`](#key-callable-execute_complete_mspi_research_pipeline)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Input Data Structure](#input-data-structure)
- [Usage](#usage)
- [Output Structure](#output-structure)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Contributing](#contributing)
- [Recommended Extensions](#recommended-extensions)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Introduction

This project provides a Python implementation of the analytical framework presented in Schmitt (2026). The core of this repository is the iPython Notebook `measuring_market_stress_with_machine_learning_draft.ipynb`, which contains a comprehensive suite of functions to replicate the paper's findings. The pipeline addresses the critical challenge of **real-time market stress monitoring** in modern algorithmic markets, where fragility manifests in the cross-section of returns and trading activity.

The paper proposes the **Market Stress Probability Index (MSPI)**, a forward-looking probability measure constructed from interpretable cross-sectional fragility signals. This codebase operationalizes the proposed solution:
-   **Validates** data integrity using strict schema checks and temporal consistency enforcement.
-   **Engineers** cross-sectional fragility features (moments, tails, liquidity) from high-frequency daily data.
-   **Learns** a sparse probability mapping using Lasso-Logit in a strict real-time expanding window.
-   **Evaluates** performance via proper scoring rules (Brier, Log Loss), calibration diagnostics, and robustness horse races against nonlinear learners (Random Forest, Gradient Boosting).

## Theoretical Background

The implemented methods combine techniques from Financial Econometrics, Machine Learning, and Probabilistic Forecasting.

**1. Cross-Sectional Fragility Signals:**
Stress leaves a footprint in the cross-section of returns. The model aggregates daily statistics into monthly features $X_t$:
$$ Z_t \equiv \frac{1}{D_t} \sum_{d \in t} z_d $$
where $z_d$ includes dispersion $\sigma^{xs}_d$, skewness, kurtosis, and tail participation $\text{Frac}^{dn}_d$.

**2. Latent Stress Definition:**
A month is labeled as stress ($S_t=1$) if market returns crash or realized volatility spikes relative to a dynamic history:
$$ S_t \equiv \mathbb{I} \{ R^{mkt}_t \le c_R \} \lor \mathbb{I} \{ \sigma^{mkt}_t \ge q_{t-1}(\alpha) \} $$

**3. Sparse Probability Modeling (Lasso-Logit):**
The probability of future stress is modeled using L1-regularized logistic regression to ensure parsimony and interpretability:
$$ MSPI_t \equiv \Pr(Y_{t+1} = 1 | X_t) = \Lambda(\beta_0 + X'_t\beta) $$
$$ (\hat{\beta}_0, \hat{\beta}) \in \arg\min_{\beta_0,\beta} \left\{ - \ell(\beta) + \lambda\|\beta\|_1 \right\} $$

**4. Real-Time Discipline:**
The pipeline enforces a strict **expanding window** protocol. At time $t$, the model is trained only on information available up to $t$, preventing look-ahead bias in both feature engineering (standardization) and model estimation.

Below is a diagram which summarizes the proposed approach:

<div align="center">
  <img src="https://github.com/chirindaopensource/measuring_market_stress_with_machine_learning/blob/main/measuring_market_stress_with_machine_learning_ipo_main.png" alt="MSPI System Architecture" width="100%">
</div>

## Features

The provided iPython Notebook (`measuring_market_stress_with_machine_learning_draft.ipynb`) implements the full research pipeline, including:

-   **Modular, Multi-Task Architecture:** The pipeline is decomposed into 22 distinct, modular tasks, each with its own orchestrator function.
-   **Configuration-Driven Design:** All study parameters (grids, splits, hyperparameters) are managed in an external `config.yaml` file.
-   **Rigorous Data Validation:** A multi-stage validation process checks schema integrity, temporal monotonicity, and return plausibility.
-   **Deterministic Execution:** Enforces reproducibility through seed control, strict causality checks, and frozen parameter sets.
-   **Comprehensive Audit Logging:** Generates detailed logs of every processing step, including invariant checks and benchmark comparisons.
-   **Reproducible Artifacts:** Generates structured results containing raw time-series, aggregated metrics, and economic analysis outputs.

## Methodology Implemented

The core analytical steps directly implement the methodology from the paper:

1.  **Configuration & Validation (Task 1):** Loads and validates the study configuration, enforcing parameter constraints and reproduction modes.

2.  **Data Ingestion & Cleansing (Tasks 2-5):** Validates CRSP micro and macro schema, enforces strict monotonicity, handles missingness, and aligns calendars.

3.  **Universe Construction (Task 6):** Filters for common shares on major exchanges with price > $1.

4.  **Feature Engineering (Tasks 7-10):** Computes daily cross-sectional moments, tail measures, and trading proxies, aggregating them to monthly frequency.

5.  **Target Construction (Tasks 11-12):** Computes market aggregates and constructs the binary stress label $S_t$ using dynamic volatility quantiles.

6.  **Standardization (Task 13):** Implements expanding-window z-scoring to prevent data leakage.

7.  **Hyperparameter Tuning (Task 14):** Optimizes regularization parameters ($\lambda$) using time-series cross-validation in the initial window.

8.  **Forecasting (Tasks 15-16):** Generates out-of-sample probability forecasts for MSPI (Lasso) and the Benchmark (Ridge).

9.  **Robustness (Task 18):** Runs a horse race against Random Forest and Gradient Boosting models with real-time Platt calibration.

10. **Evaluation (Tasks 19-21):** Computes AUC, PR-AUC, Brier Score, Log Loss, ECE, and performs block-bootstrap inference.

11. **Economic Analysis (Task 22):** Estimates predictive regressions for volatility and impulse response functions via local projections.

12. **Final Orchestration (Task 22-Plus):** Manages the end-to-end flow and enforces anti-look-ahead safety checks.

## Core Components (Notebook Structure)

The notebook is structured as a logical pipeline with modular orchestrator functions for each of the 22 major tasks. All functions are self-contained, fully documented with type hints and docstrings, and designed for professional-grade execution.

## Key Callable: `execute_complete_mspi_research_pipeline`

The project is designed around a single, top-level user-facing interface function:

-   **`execute_complete_mspi_research_pipeline`:** This master orchestrator function runs the entire automated research pipeline from end-to-end. A single call to this function reproduces the entire computational portion of the project, managing data flow between validation, feature engineering, modeling, evaluation, and economic analysis modules.

## Prerequisites

-   Python 3.9+
-   Core dependencies: `pandas`, `numpy`, `scipy`, `pyyaml`, `scikit-learn`, `statsmodels`.

## Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/chirindaopensource/measuring_market_stress_with_machine_learning.git
    cd measuring_market_stress_with_machine_learning
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Python dependencies:**
    ```sh
    pip install pandas numpy scipy pyyaml scikit-learn statsmodels
    ```

## Input Data Structure

The pipeline requires two primary DataFrames:

1.  **`df_crsp_daily` (Micro-Data)**:
    -   `PERMNO` (Int64): Security identifier.
    -   `DATE` (datetime64[ns]): Trading date.
    -   `SHRCD`, `EXCHCD` (Int64): Share and exchange codes.
    -   `PRC`, `RET`, `VOL`, `SHROUT` (float64): Price, return, volume, shares outstanding.

2.  **`df_crsp_index` (Macro-Data)**:
    -   `DATE` (datetime64[ns]): Trading date.
    -   `vwretd` (float64): Value-weighted market return.

*Note: The pipeline includes a synthetic data generator for testing purposes if access to CRSP is unavailable.*

## Usage

The notebook provides a complete, step-by-step guide. The primary workflow is to execute the final cell, which demonstrates how to use the top-level `execute_complete_mspi_research_pipeline` orchestrator:

```python
# Final cell of the notebook

# This block serves as the main entry point for the entire project.
if __name__ == '__main__':
    # 1. Load the master configuration from the YAML file.
    # (Assumes config.yaml is in the working directory)
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # 2. Load raw datasets (Example using synthetic generator provided in the notebook)
    # In production, load from CSV/Parquet: pd.read_csv(...)
    df_daily, df_index = generate_synthetic_crsp_data()

    # 3. Execute the entire replication study.
    results = execute_complete_mspi_research_pipeline(df_daily, df_index, config)
    
    # 4. Access results
    print(results["mspi_forecasts"].head())
```

## Output Structure

The pipeline returns a dictionary containing:
-   **`mspi_forecasts`**: DataFrame of out-of-sample stress probabilities ($MSPI_t$) and targets.
-   **`discrimination_metrics`**: DataFrame comparing AUC and PR-AUC across models.
-   **`probability_metrics`**: DataFrame with Brier Score, Log Loss, and ECE.
-   **`economic_analysis`**: Dictionary containing predictive regression stats and IRF DataFrames.
-   **`audit_log`**: A detailed record of data cleansing stats, universe counts, and hyperparameter choices.

## Project Structure

```
measuring_market_stress_with_machine_learning/
│
├── measuring_market_stress_with_machine_learning_draft.ipynb   # Main implementation notebook
├── config.yaml                                                 # Master configuration file
├── requirements.txt                                            # Python package dependencies
│
├── LICENSE                                                     # MIT Project License File
└── README.md                                                   # This file
```

## Customization

The pipeline is highly customizable via the `config.yaml` file. Users can modify study parameters such as:
-   **Stress Definition:** `vol_quantile_alpha` (e.g., 0.90 or 0.95), `return_cutoff_c_R`.
-   **Feature Engineering:** `tail_threshold_tau` (e.g., 0.05).
-   **Learning Protocol:** `initial_training_window_months`, `cv_n_splits`.
-   **Models:** Hyperparameter grids for Lasso, Ridge, RF, and GB.

## Contributing

Contributions are welcome. Please fork the repository, create a feature branch, and submit a pull request with a clear description of your changes. Adherence to PEP 8, type hinting, and comprehensive docstrings is required.

## Recommended Extensions

Future extensions could include:
-   **Alternative Stress Definitions:** Incorporating liquidity or credit spreads into the stress label.
-   **Intraday Features:** Using high-frequency TAQ data for realized measures.
-   **Deep Learning Models:** Benchmarking against LSTM or Transformer architectures (with proper calibration).
-   **International Markets:** Extending the analysis to global equity markets.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation

If you use this code or the methodology in your research, please cite the original paper:

```bibtex
@article{schmitt2026algorithmic,
  title={Algorithmic Monitoring: Measuring Market Stress with Machine Learning},
  author={Schmitt, Marc},
  journal={arXiv preprint arXiv:2602.07066},
  year={2026}
}
```

For the implementation itself, you may cite this repository:
```
Chirinda, C. (2026). Algorithmic Monitoring: Measuring Market Stress with Machine Learning: An Open Source Implementation.
GitHub repository: https://github.com/chirindaopensource/measuring_market_stress_with_machine_learning
```

## Acknowledgments

-   Credit to **Marc Schmitt** for the foundational research that forms the entire basis for this computational replication.
-   This project is built upon the exceptional tools provided by the open-source community. Sincere thanks to the developers of the scientific Python ecosystem, including **Pandas, NumPy, SciPy, Scikit-Learn, and Statsmodels**.

--

*This README was generated based on the structure and content of the `measuring_market_stress_with_machine_learning_draft.ipynb` notebook and follows best practices for research software documentation.*
