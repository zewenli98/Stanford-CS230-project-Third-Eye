import argparse
import pandas as pd
import matplotlib.pyplot as plt


def plot_loss(metrics_file: str):
    # read the metrics file
    df = pd.read_csv(metrics_file)
    # plot the train loss first 1071 steps
    plt.plot(df["train_loss"][:1047], label="train loss")
    # plt.plot(df["train_loss"], label="train loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    model_name = metrics_file.split('/')[0]
    plt.title(f"Train Loss - {model_name}")
    plt.legend()
    plt.savefig(f"{model_name}_train_loss.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", "-m", type=str, required=True)
    args = parser.parse_args()
    plot_loss(args.metrics)
