import argparse
from train import train
from evaluate import evaluate

def main():
    parser = argparse.ArgumentParser(description="ASHT-KD Place Recognition")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate'], help="Choose train or evaluate mode")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the dataset directory")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to load a model checkpoint for evaluation")

    args = parser.parse_args()

    if args.mode == 'train':
        print("Starting training...")
        train(data_dir=args.data_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    elif args.mode == 'evaluate':
        print("Starting evaluation...")
        evaluate(data_dir=args.data_dir, batch_size=args.batch_size, checkpoint=args.checkpoint)

if __name__ == "__main__":
    main()
