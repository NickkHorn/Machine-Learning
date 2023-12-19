import numpy as np

model = Adaline(N_FEATURES)
losses, accs = model.fit(X, Y)

plt.plot([i+1 for i in range(len(losses))], losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss function value", loc="center")
plt.show()

plt.plot([i+1 for i in range(len(accs))], accs)
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy", loc="center")
plt.show()