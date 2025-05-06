
DATASET_PREFIX: str = ""  # "" for local paths, "/kaggle/input/nlp-tagger-data/" for kaggle

WORD_VECTOR_SIZE: int = 50
GLOBAL_RANDOM_SEED: int = 42

from model.mlp_tagger_char_cnn import MlpTaggerCharCnn
from model.mlp_tagger_simple import MlpTaggerSimple
from model.mlp_tagger_subwords import MlpTaggerSubwords

FILE_LOCATIONS = {
    "pos": {
        "training": f"{DATASET_PREFIX}data/pos/train",
        "development": f"{DATASET_PREFIX}data/pos/dev",
        "test": f"{DATASET_PREFIX}data/pos/test",
    },
    "ner": {
        "training": f"{DATASET_PREFIX}data/ner/train",
        "development": f"{DATASET_PREFIX}data/ner/dev",
        "test": f"{DATASET_PREFIX}data/ner/test",
    },
    "embeddings": {
        "vocabulary": f"{DATASET_PREFIX}data/embeddings/vocab.txt",
        "vectors": f"{DATASET_PREFIX}data/embeddings/wordVectors.txt",
    },
}

HYPER_PARAMETER_SETS = {
    "pos": {
        # "embedding_learning_rate": 1e-2,
        # "embedding_weight_decay": 0,
        "classifier_learning_rate": 1e-2,
        "classifier_weight_decay": 3e-4,
        "scheduler_patience": 1,
        "scheduler_decay_factor": 0.1,
        "scheduler_min_lr": 1e-4,
        "training_epochs": 50,
        "number_of_warm_epochs": 6,
        "classifier_hidden_units": 64,
        "dropout_probability": 0.25,
        "batch_size": int(8192 * 4),
    },
    "ner": {
        # "embedding_learning_rate": 6e-1,
        # "embedding_weight_decay": 0.0,
        # "classifier_learning_rate": 3e-4,
        "classifier_learning_rate": 3e-2,
        "classifier_weight_decay": 3e-4,
        "scheduler_patience": 1,
        "scheduler_decay_factor": 0.1,
        "scheduler_min_lr": 1e-5,
        "training_epochs": 40,
        "number_of_warm_epochs": 0,
        "classifier_hidden_units": 128,
        "dropout_probability": 0.5,
        "batch_size": int(8192),
    },
    "part-1-pos": {
        "classifier_learning_rate": 1.5e-2,
        "classifier_weight_decay": 1e-4,
        "scheduler_patience": 0,
        "scheduler_decay_factor": 0.4,
        "scheduler_min_lr": 1e-3,
        "training_epochs": 6,
        "classifier_hidden_units": 184,
        "dropout_probability": 0.25,
        "batch_size": int(8192 * 4),
    },
    "part-1-ner": {
        "classifier_learning_rate": 34e-2,
        "classifier_weight_decay": 4e-4,
        "scheduler_patience": 0,
        "scheduler_decay_factor": 0.3,
        "scheduler_min_lr": 1e-5,
        "training_epochs": 11,
        "classifier_hidden_units": 90,
        "dropout_probability": 0.25,
        "batch_size": int(8192),
    },
    "part-3-pos": {
        "classifier_learning_rate": 1.5e-2,
        "classifier_weight_decay": 1e-4,
        "scheduler_patience": 0,
        "scheduler_decay_factor": 0.4,
        "scheduler_min_lr": 1e-3,
        "training_epochs": 10,
        "classifier_hidden_units": 184,
        "dropout_probability": 0.25,
        "batch_size": int(8192 * 4),
    },
    "part-3-ner": {
        "classifier_learning_rate": 3e-2,
        "classifier_weight_decay": 3e-4,
        "scheduler_patience": 1,
        "scheduler_decay_factor": 0.1,
        "scheduler_min_lr": 1e-5,
        "training_epochs": 40,
        "classifier_hidden_units": 128,
        "dropout_probability": 0.55,
        "batch_size": int(8192*4),
    },
    "part-4-pos": {
        "classifier_learning_rate": 1e-2,
        "classifier_weight_decay": 3e-4,
        "scheduler_patience": 1,
        "scheduler_decay_factor": 0.1,
        "scheduler_min_lr": 1e-4,
        "training_epochs": 27,
        "classifier_hidden_units": 64,
        "dropout_probability": 0.25,
        "batch_size": int(8192 * 4),
    },

    "part-4-ner": {
        "classifier_learning_rate": 3e-2,
        "classifier_weight_decay": 3e-4,
        "scheduler_patience": 1,
        "scheduler_decay_factor": 0.1,
        "scheduler_min_lr": 3e-3,
        "training_epochs": 40,
        "classifier_hidden_units": 128,
        "dropout_probability": 0.55,
        "batch_size": int(8192*2),
    },

    "part-5-pos": {
        "classifier_learning_rate": 3e-2,
        "classifier_weight_decay": 3e-4,
        "scheduler_patience": 1,
        "scheduler_decay_factor": 0.1,
        "scheduler_min_lr": 1e-5,
        "training_epochs": 11,
        "number_of_warm_epochs": 0,
        "classifier_hidden_units": 128,
        "dropout_probability": 0.25,
        "batch_size": int(8192),
    },
    "part-5-ner": {
        "classifier_learning_rate": 3e-2,
        "classifier_weight_decay": 3e-4,
        "scheduler_patience": 1,
        "scheduler_decay_factor": 0.1,
        "scheduler_min_lr": 3e-3,
        "training_epochs": 40,
        "classifier_hidden_units": 128,
        "dropout_probability": 0.25,
        "batch_size": int(8192*2),
    },
}
MODEL_TYPE_TO_CLASS = {
    "simple": MlpTaggerSimple,
    "subword": MlpTaggerSubwords,
    "charcnn": MlpTaggerCharCnn
}
