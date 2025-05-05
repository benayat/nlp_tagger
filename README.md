# README

## Project Overview
This project implements a neural network-based model for sequence tagging tasks such as Part-of-Speech (POS) tagging and Named Entity Recognition (NER). The framework supports multiple architectures, including simple feedforward models, subword-based models, and character-level CNNs.

## Requirements
- Python 3.12
- PyTorch
- NumPy
- Other dependencies listed in `requirements.txt`

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/benayat/your-repo-name.git
   cd your-repo-name
   ```
2. Create a virtual environment and activate it:
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Training
Run the `main.py` script to train the model. Use the following arguments to configure the task and architecture:
- `--task`: Choose between `pos` (Part-of-Speech tagging) or `ner` (Named Entity Recognition). Default: `pos`.
- `--architecture`: Choose the model architecture: `simple`, `subword`, or `charcnn`. Default: `simple`.
- `--pretrained`: Use pretrained embeddings (optional).
- `--char-max-len`: Maximum character length for character-level CNNs. Default: `15`.

Example:
```bash
python main.py --task pos --architecture charcnn --pretrained
```

### Output
- Training and validation metrics (loss and accuracy) are printed for each epoch.
- Predictions for the test set are saved in a file named `test.<task>` (e.g., `test.pos` or `test.ner`).

## Configuration
Hyperparameters such as learning rate, batch size, and training epochs are defined in `constants/constants.py` under `HYPER_PARAMETER_SETS`.

## Results
Training and validation results, including loss and accuracy, are logged during training. Example:
```
Fine 01 | train 0.46543 / 86.76% | dev 0.53345 / 84.18% | lr 3.0e-02
Fine 02 | train 0.42349 / 88.16% | dev 0.55870 / 82.97% | lr 3.0e-02
...
```

## Notes
- The model automatically adjusts learning rate and weight decay using a scheduler.
- Ensure the required data files (embeddings, training, development, and test sets) are placed in the paths specified in `constants/constants.py`.
