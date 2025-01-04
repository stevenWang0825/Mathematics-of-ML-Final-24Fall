import torch
import matplotlib.pyplot as plt

def plot_metrics(metrics_file="metrics.pt", step_interval=20):
    """
    从 metrics_file 中读取训练过程的 loss/acc，间隔 step_interval 个 step 画一个点。
    横坐标使用对数刻度。
    """
    # 读取字典格式的训练数据
    data = torch.load(metrics_file)

    # 原始数据
    train_losses = data["train_losses"]
    train_accs   = data["train_accs"]
    val_losses   = data["val_losses"]
    val_accs     = data["val_accs"]

    # 每隔 step_interval 取一次数据
    sampled_indices = range(0, len(train_losses), step_interval)
    train_losses_sampled = [train_losses[i] for i in sampled_indices]
    train_accs_sampled   = [train_accs[i]   for i in sampled_indices]
    val_losses_sampled   = [val_losses[i]   for i in sampled_indices]
    val_accs_sampled     = [val_accs[i]     for i in sampled_indices]
    sampled_epochs       = [i + 1 for i in sampled_indices]

    # 创建图像和子图
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # -- Loss 曲线子图 --
    axs[0].plot(sampled_epochs, train_losses_sampled, label="Train Loss")
    axs[0].plot(sampled_epochs, val_losses_sampled,   label="Valid Loss")
    axs[0].set_xscale("log")  # 对数坐标
    axs[0].set_xlabel("Epoch (log scale)")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].set_title("Loss vs. Epoch")

    # -- Accuracy 曲线子图 --
    axs[1].plot(sampled_epochs, train_accs_sampled, label="Train Acc")
    axs[1].plot(sampled_epochs, val_accs_sampled,   label="Valid Acc")
    axs[1].set_xscale("log")  # 对数坐标
    axs[1].set_xlabel("Epoch (log scale)")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()
    axs[1].set_title("Accuracy vs. Epoch")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 根据需要调整 step_interval 或者 metrics_file 路径
    plot_metrics("metrics.pt", step_interval=1)