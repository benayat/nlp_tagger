from __future__ import annotations
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from constants.constants import GLOBAL_RANDOM_SEED, MODEL_TYPE_TO_CLASS, HYPER_PARAMETER_SETS, FILE_LOCATIONS
from model.utils import execute_epoch, build_data_loader, visualize_model_results
from reader.ner_reader import NerReader
from reader.pos_reader import PosReader
from reader.utils import load_text_embeddings

if __name__ == "__main__":
    torch.manual_seed(GLOBAL_RANDOM_SEED)
    np.random.seed(GLOBAL_RANDOM_SEED)
    random.seed(GLOBAL_RANDOM_SEED)
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["pos", "ner"], default="pos")
    parser.add_argument("--architecture", choices=["simple", "subword", "charcnn"], default="simple")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--hyper_param_group", choices=["part-1", "part-3","part-4","part-5"], default="part-1")
    args = parser.parse_args()

    COMPUTE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TASK: str = args.task
    CONFIG = HYPER_PARAMETER_SETS[f"{args.hyper_param_group}-{TASK}"]

    # 1. Prepare embeddings and reader
    pretrained_dictionary = load_text_embeddings(
        FILE_LOCATIONS["embeddings"]["vocabulary"],
        FILE_LOCATIONS["embeddings"]["vectors"]
    ) if args.pretrained else None

    ReaderClass = PosReader if TASK == "pos" else NerReader
    data_reader = ReaderClass(
        FILE_LOCATIONS[TASK]["training"],
        FILE_LOCATIONS[TASK]["development"],
        FILE_LOCATIONS[TASK]["test"],
        pretrained_dictionary,
        use_subwords_in_reader=args.architecture == "subword",
        use_charcnn=args.architecture == "charcnn",
    )

    # 2. Create data loaders

    loader_train = build_data_loader(data_reader=data_reader, batch_size=CONFIG['batch_size'], split_label="train",
                                     shuffle_flag=True)
    loader_dev = build_data_loader(data_reader=data_reader, batch_size=CONFIG['batch_size'], split_label="dev",
                                   shuffle_flag=False)

    # 3. Build model instance
    model = MODEL_TYPE_TO_CLASS[args.architecture]
    model_instance = model(
        reader=data_reader,
        hidden_units=CONFIG["classifier_hidden_units"],
        dropout_probability=CONFIG["dropout_probability"]
    ).to(COMPUTE_DEVICE)

    loss_function = nn.CrossEntropyLoss(
        ignore_index=data_reader.tag_to_index[data_reader.padding_tag]
    )
    optimizer = torch.optim.Adam(model_instance.parameters(), lr=CONFIG["classifier_learning_rate"],
                                 weight_decay=CONFIG["classifier_weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=CONFIG["scheduler_patience"],
        factor=CONFIG["scheduler_decay_factor"],
        min_lr=CONFIG["scheduler_min_lr"]
    )
    previous_learning_rate_value = scheduler.get_last_lr()[0]

    print("\n=== Training: ===")
    dev_accuracies = []
    dev_losses = []
    for epoch_id in range(1, CONFIG["training_epochs"] + 1):
        train_loss, train_accuracy = execute_epoch(
            model_instance, loader_train, loss_function, optimizer, reader=data_reader,
            is_training=True,
            device=COMPUTE_DEVICE
        )
        dev_loss, dev_accuracy = execute_epoch(
            model_instance, loader_dev, loss_function, None,
            reader=data_reader, is_training=False, device=COMPUTE_DEVICE
        )
        # Save dev accuracy and loss for visualization later
        dev_accuracies.append(dev_accuracy)
        dev_losses.append(dev_loss)

        scheduler.step(dev_loss)
        current_learning_rate_value = scheduler.get_last_lr()[0]
        if current_learning_rate_value != previous_learning_rate_value:
            optimizer.param_groups[0]["weight_decay"] *= CONFIG["scheduler_decay_factor"]
            previous_learning_rate_value = current_learning_rate_value
            print(f"Learning-rate decayed → {current_learning_rate_value:.1e}; "
                  f"weight-decay scaled × {CONFIG["scheduler_decay_factor"]:.3f}")

        print(f"Fine {epoch_id:02d} | "
              f"train {train_loss:.5f} / {train_accuracy * 100:5.2f}% | "
              f"dev {dev_loss:.5f} / {dev_accuracy * 100:5.2f}% | "
              f"lr {current_learning_rate_value:.1e}")

    # ─── Inference on test set ───
    test_windows_list = list(data_reader.sliding_windows(
        FILE_LOCATIONS[TASK]["test"], test_mode=True)
    )
    test_tensor_input = torch.tensor(test_windows_list)
    logits_test = model_instance(test_tensor_input.to(COMPUTE_DEVICE))
    predicted_tag_indices = logits_test.argmax(1).cpu().tolist()

    output_folder = f"{args.hyper_param_group}"
    os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist

    output_filename = fr"{output_folder}/test.{TASK}"
    with open(output_filename, "w", encoding="utf-8") as file:
        for index in predicted_tag_indices:
            file.write(data_reader.index_to_tag[index] + "\n")
    print(f"\nPredictions stored in file: {output_filename}")
    # Visualize the model results
    visualize_model_results(num_epochs=CONFIG["training_epochs"], dev_accuracies=dev_accuracies,dev_losses=dev_losses, task=TASK, part=args.hyper_param_group)

    # free cuda memory
    import  gc
    torch.cuda.empty_cache()
    gc.collect()