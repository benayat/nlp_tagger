import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from data_reader.ner_reader import NERReader
from data_reader.pos_reader import POSReader
from data_reader.util import load_dataset_from_reader, load_test_dataset_from_reader, \
    create_word_to_vec_vocab_dict_from_embeddings_files
from model.mlp_tagger import MLPTagger
from model.util import create_optimizer_and_scheduler, train_one_epoch, evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
vendor_prefix = ''
POS_TRAIN_PATH = 'data/pos/train'
NER_TRAIN_PATH = 'data/ner/train'
POS_DEV_PATH = 'data/pos/dev'
NER_DEV_PATH = 'data/ner/dev'
WORDS_PATH = 'data/embeddings/vocab.txt'
VECS_PATH = 'data/embeddings/wordVectors.txt'
task = 'ner'
word_to_vec = create_word_to_vec_vocab_dict_from_embeddings_files(WORDS_PATH, VECS_PATH)
reader = POSReader(train_file_path=POS_TRAIN_PATH, dev_file_path=POS_DEV_PATH, word_to_vec=word_to_vec) if task == 'pos' else NERReader(train_file_path=NER_TRAIN_PATH, dev_file_path=NER_DEV_PATH, word_to_vec=word_to_vec)

X_train, y_train = load_dataset_from_reader(reader, "train")
X_val, y_val = load_dataset_from_reader(reader, "dev") if task == 'pos' else load_dataset_from_reader(reader, "dev")

pos_params = {
    'initial_lr': 1.5e-3, 'weight_decay':1e-4,'scheduler_patience': 2, 'scheduler_factor': 0.4, 'min_lr': 1e-3, 'epochs': 8, 'hidden_dim': 184
}
ner_params = {
    'initial_lr': 1e-3, 'weight_decay':5e-4,'scheduler_patience': 2, 'scheduler_factor': 0.35, 'min_lr': 1e-5, 'epochs': 50, 'hidden_dim': 100
}
params = pos_params if task == 'pos' else ner_params

# Create data loaders
batch_size = 256
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

# Model setup
vocab_size = len(reader.vocab)
pad_idx = reader.word_to_index[reader.pad_token]
output_dim = len(reader.tag_vocab)
pretrained_embeddings = torch.tensor(reader.embeddings_matrix)

model = MLPTagger(
    vocab_size=vocab_size,
    embedding_dim=50,
    hidden_dim=params['hidden_dim'],
    output_dim=output_dim,
    pad_idx=pad_idx,
    pretrained_embeddings=pretrained_embeddings,
).to(device)

dev_losses = []
dev_accuracies = []

pad_idx = reader.tag_to_index.get(reader.pad_tag, -100)
o_idx = reader.tag_to_index.get("O")
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer, scheduler = create_optimizer_and_scheduler(model, params['initial_lr'], params['weight_decay'], params['scheduler_patience'], params['scheduler_factor'], params['min_lr'])
pad_idx = reader.tag_to_index.get(reader.pad_tag, -100)
o_idx   = reader.tag_to_index.get("O") if isinstance(reader, NERReader) else None
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

for epoch in range(params['epochs']):
    train_loss, train_acc = train_one_epoch(
        model, train_loader,  criterion, optimizer, pad_idx, o_idx, device)
    val_loss,   val_acc   = evaluate(
        model, val_loader, criterion, pad_idx, o_idx, device)
    dev_losses.append(val_loss)
    dev_accuracies.append(val_acc)
    scheduler.step(val_loss)
    print(f"Ep {epoch+1:02d} | train loss: {train_loss:.4f} train accuracy:{train_acc:.4f} | "
          f"val loss: {val_loss:.4f} val accuracy: {val_acc:.4f} | lr: {scheduler.get_last_lr()[0]:.1e}")

# predict on the test set
test_path = 'data/pos/test' if task == 'pos' else 'data/ner/test'
test_loader = DataLoader(load_test_dataset_from_reader(reader, test_path), batch_size=batch_size)

predictions = []
for X_batch in test_loader:
    X_batch = X_batch.to(device)
    logits = model(X_batch)
    preds = logits.argmax(dim=1)
    predictions.append(preds.cpu())
with open(f'test1.{task}', 'w') as f:
    for batch in predictions:
        for pred in batch:
            tag = reader.get_tag_from_index(pred.item())  # Convert index to tag
            f.write(f"{tag}\n")
        f.write("\n")