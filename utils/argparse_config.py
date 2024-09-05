import argparse

def get_args():
    parser = argparse.ArgumentParser(description="ASHT-KD Place Recognition Framework")

    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate'],
                        help="Mode: train or evaluate (default: train)")
    parser.add_argument('--data_dir', type=str, required=True,
                        help="Path to the dataset directory")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help="Path to load a model checkpoint for evaluation or resume training")

    parser.add_argument('--epochs', type=int, default=10,
                        help="Number of epochs to train (default: 10)")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="Batch size for training or evaluation (default: 16)")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="Learning rate for the optimizer (default: 0.001)")
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help="Weight decay for the optimizer (default: 1e-5)")
    parser.add_argument('--temperature', type=float, default=3.0,
                        help="Temperature for knowledge distillation loss (default: 3.0)")

    parser.add_argument('--eval_batch_size', type=int, default=16,
                        help="Batch size for evaluation (default: 16)")

    parser.add_argument('--gpu', type=int, default=0,
                        help="GPU id to use (default: 0)")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    print(args)
