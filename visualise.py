import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("logs/pretrain_metrics.csv")

plt.figure(figsize=(10,5))
plt.plot(df["epoch"], df["loss"], marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Pretraining Loss")
plt.grid(True)
plt.show()

plt.figure(figsize=(10,5))
plt.plot(df["epoch"], df["psnr"], marker="o")
plt.xlabel("Epoch")
plt.ylabel("PSNR")
plt.title("Pretraining PSNR")
plt.grid(True)
plt.show()