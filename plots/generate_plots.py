import pandas as pd
from matplotlib import pyplot as plt

# Validation Jaccard Index per beta

f, ax = plt.subplots(figsize=(8, 8))
ax.set(xscale="log")
df = pd.read_csv('jaccard_indices_per_beta.csv')
df.plot(x='beta', ax=ax)
plt.ylabel('jaccard index')
plt.title('Validation Jaccard Index per beta')
plt.savefig('jaccard_indices_per_beta.pdf')

# Validation Loss per model

f, ax = plt.subplots(figsize=(8, 8))
ax.set(yscale="log")
df = pd.read_csv('/data/plots/validation_losses.csv')
df.plot(x='epoch', ax=ax)
plt.title('Validation Loss per model')
plt.ylabel('loss')
plt.savefig('/data/plots/validation_losses.pdf')

# Validation Loss per model

f, ax = plt.subplots(figsize=(8, 8))
ax.set(yscale="log")
df = pd.read_csv('/data/plots/train_losses.csv')
df.plot(x='epoch', ax=ax)
plt.title('Validation Loss per model')
plt.ylabel('loss')
plt.savefig('train_losses.pdf')
