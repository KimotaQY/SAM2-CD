import matplotlib.pyplot as plt


def visualize_batch(images_a, images_b, masks):
    input_A_np = images_a.cpu().numpy().transpose(0, 2, 3, 1)
    input_B_np = images_b.cpu().numpy().transpose(0, 2, 3, 1)
    mask_np = masks.cpu().numpy().transpose(0, 2, 3, 1)

    # 可视化前后时相及其对应的mask
    for i in range(min(4, images_a.size(0))):  # 打印前4个图像样本
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow((input_A_np[i] * 0.5) + 0.5)  # 反归一化
        axs[0].set_title("Image A (T1)")
        axs[0].axis("off")

        axs[1].imshow((input_B_np[i] * 0.5) + 0.5)  # 反归一化
        axs[1].set_title("Image B (T2)")
        axs[1].axis("off")

        axs[2].imshow(mask_np[i].squeeze(), cmap="gray")
        axs[2].set_title("Mask")
        axs[2].axis("off")

        plt.show()


def visualize_batch_test(images_a, images_b, masks, pred, **kwargs):
    input_A_np = images_a.cpu().numpy().transpose(0, 2, 3, 1)
    input_B_np = images_b.cpu().numpy().transpose(0, 2, 3, 1)
    # mask_np = masks.cpu().numpy().transpose(0, 2, 3, 1)
    # pred_np = pred.cpu().numpy().transpose(0, 2, 3, 1)
    mask_np = masks.transpose(0, 2, 3, 1)
    pred_np = pred.transpose(0, 2, 3, 1)

    plt.title(f"F1: {kwargs['F1']}, IoU: {kwargs['IoU']}")

    # 可视化前后时相及其对应的mask
    for i in range(min(4, images_a.size(0))):  # 打印前4个图像样本
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))

        axs[0, 0].imshow((input_A_np[i] * 0.5) + 0.3)  # 反归一化
        axs[0, 0].set_title("Image A (T1)")
        axs[0, 0].axis("off")

        axs[0, 1].imshow((input_B_np[i] * 0.5) + 0.3)  # 反归一化
        axs[0, 1].set_title("Image B (T2)")
        axs[0, 1].axis("off")

        axs[1, 0].imshow(mask_np[i].squeeze(), cmap="gray")
        axs[1, 0].set_title("Mask")
        axs[1, 0].axis("off")

        axs[1, 1].imshow(pred_np[i].squeeze(), cmap="gray")
        axs[1, 1].set_title("Pred")
        axs[1, 1].axis("off")

        plt.show()
