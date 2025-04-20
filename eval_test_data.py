from torch.utils.data import DataLoader
import torch

from data_reader.ner_reader import NERReader
from data_reader.pos_reader import POSReader
from data_reader.util import load_test_dataset_from_reader
from model.mlp_tagger import MLPTagger
from model.util import predict

task= 'pos'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
vendor_prefix = ''
POS_TRAIN_PATH = 'data/pos/train'
NER_TRAIN_PATH = 'data/ner/train'
POS_DEV_PATH = 'data/pos/dev'
NER_DEV_PATH = 'data/ner/dev'
reader = POSReader(POS_TRAIN_PATH) if task == 'pos' else NERReader(NER_TRAIN_PATH)
batch_size = 16384

test_path = f'data/{task}/test'
test_loader = DataLoader(load_test_dataset_from_reader(reader, test_path), batch_size=batch_size)

model_path = f'models/{task}_model.pth'
vocab_size = len(reader.vocab)
pad_idx = reader.word_to_index[reader.pad_token]
output_dim = len(reader.tag_vocab)
model = MLPTagger(
    vocab_size=vocab_size,
    embedding_dim=50,
    hidden_dim=184 if task == 'pos' else 100,
    output_dim=output_dim,
    pad_idx=pad_idx,
).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
predictions = []
for X_batch in test_loader:
    predictions.append(predict(model, X_batch, device))
with open(f'test1.{task}', 'w') as f:
    for batch_idx, batch in enumerate(predictions):
        for pred_idx, pred in enumerate(batch):
            tag = reader.get_tag_from_index(pred.item())  # index to tag
            f.write(f"{tag}")
            if not (batch_idx == len(predictions) - 1 and pred_idx == len(batch) - 1):
                f.write("\n")