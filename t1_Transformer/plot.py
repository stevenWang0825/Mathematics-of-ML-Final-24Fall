import torch
import matplotlib.pyplot as plt

def plot_metrics(metrics_file="metrics.pt", step_interval=20):
    """
    从 metrics_file 读出训练过程中记录的 loss/acc 并画图
    每隔 step_interval 输出一个点
    """
    data = torch.load(metrics_file)  # 里面是一个字典

    # 原始数据
    train_losses = data["train_losses"]
    train_accs = data["train_accs"]
    val_losses = data["val_losses"]
    val_accs = data["val_accs"]
    print(train_losses)
    # 按 step_interval 取样
    sampled_indices = range(0, len(train_losses), step_interval)
    train_losses_sampled = [train_losses[i] for i in sampled_indices]
    train_accs_sampled = [train_accs[i] for i in sampled_indices]
    val_losses_sampled = [val_losses[i] for i in sampled_indices]
    val_accs_sampled = [val_accs[i] for i in sampled_indices]
    sampled_epochs = [i + 1 for i in sampled_indices]

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # ---- Loss 曲线 ----
    axs[0].plot(sampled_epochs, train_losses_sampled, label="Train Loss")
    axs[0].plot(sampled_epochs, val_losses_sampled, label="Valid Loss")
    axs[0].set_xscale("log")  # 对数横坐标
    axs[0].set_xlabel("Epoch (log scale)")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].set_title("Loss vs. Epoch")

    # ---- Accuracy 曲线 ----
    axs[1].plot(sampled_epochs, train_accs_sampled, label="Train Acc")
    axs[1].plot(sampled_epochs, val_accs_sampled, label="Valid Acc")
    axs[1].set_xscale("log")  # 对数横坐标
    axs[1].set_xlabel("Epoch (log scale)")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()
    axs[1].set_title("Accuracy vs. Epoch")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_metrics("97_2_5e-5_1e6_0.7.pt", step_interval=50)