from argparse import ArgumentParser
from training import main

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--training_fraction", type=float, default=0.2)
    parser.add_argument("--prime", type=int, default=97)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim_model", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1)
    parser.add_argument("--num_steps", type=int, default=100000)
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    # 直接将其转化为字典
    main(vars(args))