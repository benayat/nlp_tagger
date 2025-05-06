# README

## Project Overview
This project implements a neural network-based model for sequence tagging tasks such as Part-of-Speech (POS) tagging and Named Entity Recognition (NER). The framework supports multiple architectures, including simple feedforward models, subword-based models, and character-level CNNs.

## To run:
1. Clone the repo:
   git clone https://github.com/benayat/nlp_tagger.git
   cd nlp_tagger
2. Create a virtual environment and activate it:
   python -m venv .venv
   source .venv/bin/activate
3. Install dependencies:
   pip install -r requirements.txt
4. Run the main.py script to train the model. Use the following arguments to configure the task and architecture:
   - `--task`: Choose between `pos` (Part-of-Speech tagging) or `ner` (Named Entity Recognition). Default: `pos`.
   - `--architecture`: Choose the model architecture: `simple`, `subword`, or `charcnn`. Default: `simple`.
   - `--pretrained`: Use pretrained embeddings (optional).
   - `--char-max-len`: Maximum character length for character-level CNNs. Default: `15`.
5. for running by part: 
- part-1: pos: `python main.py --task pos --architecture simple --hyper_param_group part-1`
- part-1: ner: `python main.py --task ner --architecture simple --hyper_param_group part-1`
- part-3: pos: `python main.py --task pos --architecture simple --pretrained --hyper_param_group part-3`
- part-3: ner: `python main.py --task ner --architecture simple --pretrained --hyper_param_group part-3`
- part-4: pos: `python main.py --task pos --architecture subword --hyper_param_group part-4`
- part-4: ner: `python main.py --task ner --architecture subword --hyper_param_group part-4`
- part-4: pos-pretrained: `python main.py --task pos --architecture subword --pretrained --hyper_param_group part-4`
- part-4: ner-pretrained: `python main.py --task ner --architecture subword --pretrained --hyper_param_group part-4`
- part-5: pos: `python main.py --task pos --architecture charcnn --hyper_param_group part-5`
- part-5: ner: `python main.py --task ner --architecture charcnn --hyper_param_group part-4`
- part-6: `python lm_script.py`

## Configuration
Hyperparameters such as learning rate, batch size, and training epochs are defined in `constants/constants.py` under `HYPER_PARAMETER_SETS`.

## Results
Training and validation results, including loss and accuracy, are logged during training.
I recorded the successful runs logs in the file `results/results.md` for reference.

## Notes
- The model automatically adjusts learning rate with a scheduler, and a manual intervention for the weight-decay adjustment. 
- Ensure the required data paths (embeddings, training, development, and test sets) are placed in the paths specified in `constants/constants.py`.
