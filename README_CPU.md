# Three-Body Project — Run on CPU

This guide explains how to run the project on a computer **without an NVIDIA GPU**.

The notebook will work on CPU, but posterior sampling may take longer.

## 1. Extract the ZIP file

Extract `three_body_share.zip`.

Open PowerShell inside the extracted `three_body_share` folder.

The folder should contain:

```text
three_body_share/
├── code/
├── data/
│   └── test.npz
├── models/
│   └── trained_model.keras
├── results/
├── results_notebook.ipynb
└── requirements.txt
```

## 2. Install Python 3.12

Python 3.12 is recommended.

Check whether it is installed:

```powershell
py -3.12 --version
```

If the command works, continue to the next step.

## 3. Create a virtual environment

Run:

```powershell
py -3.12 -m venv env_three_body
```

Activate it:

```powershell
.\env_three_body\Scripts\Activate.ps1
```

After activation, the terminal should begin with:

```text
(env_three_body)
```

If PowerShell blocks the activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\env_three_body\Scripts\Activate.ps1
```

## 4. Upgrade pip

```powershell
python -m pip install --upgrade pip
```

## 5. Install the CPU version of PyTorch

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 6. Install the remaining packages

Use this command instead of installing the CUDA version from `requirements.txt`:

```powershell
pip install bayesflow==2.0.12 numpy scipy matplotlib jupyter nbformat ipykernel
```

## 7. Check that everything works

Check PyTorch:

```powershell
python -c "import torch; print('Torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

On a computer without an NVIDIA GPU, the expected result is:

```text
CUDA available: False
```

That is normal.

Check BayesFlow:

```powershell
python -c "import os; os.environ['KERAS_BACKEND']='torch'; import bayesflow; print('BayesFlow:', bayesflow.__version__)"
```

The expected BayesFlow version is:

```text
2.0.12
```

## 8. Start Jupyter Notebook

```powershell
jupyter notebook
```

A browser window should open.

Open:

```text
results_notebook.ipynb
```

## 9. Select the correct Python kernel

In Jupyter, select the kernel connected to `env_three_body`.

If it does not appear, close Jupyter and run:

```powershell
python -m ipykernel install --user --name env_three_body --display-name "Python (env_three_body)"
```

Then start Jupyter again:

```powershell
jupyter notebook
```

Select:

```text
Python (env_three_body)
```

## 10. Run the notebook

In Jupyter, choose:

```text
Run → Run All Cells
```

The notebook should load:

- `models/trained_model.keras`
- `data/test.npz`

It should generate or update:

- `results/posterior_example.png`
- `results/posterior_predictive.png`

## Important notes

### CPU execution is slower

The model was originally trained and tested with a GPU. It can still run on CPU, but posterior sampling may take several minutes depending on the computer.

### Results may differ slightly

Posterior sampling is stochastic, so the plots may not be exactly identical on every run.

### Rare energy-drift error

A small number of posterior samples can produce a near-collision trajectory and fail the strict energy-drift check.

If the notebook raises an energy-drift `RuntimeError`:

1. Restart the notebook kernel.
2. Run all cells again.

This does not necessarily mean the installation is incorrect.

## Quick setup

Run these commands from the extracted project folder:

```powershell
py -3.12 -m venv env_three_body
.\env_three_body\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install bayesflow==2.0.12 numpy scipy matplotlib jupyter nbformat ipykernel
jupyter notebook
```

Then open `results_notebook.ipynb`, select `Python (env_three_body)`, and run all cells.
