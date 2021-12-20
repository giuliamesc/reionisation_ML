import numpy as np, matplotlib.pyplot as plt
import pickle

path = './checkpoints/'
loss, val_loss = pickle.load(open("%sloss_train" %path, "rb"))['train_loss'], pickle.load(open("%sloss_test" %path, "rb"))['test_loss']
R2, val_R2 = pickle.load(open("%sR2_train" %path, "rb"))['R2_train'], pickle.load(open("%sR2_test" %path, "rb"))['R2_test']

idx_best_mode = np.argmin(val_loss)

# Plot
fig, ax = plt.subplots(1, 2, figsize=(16,8))
ax[0].plot(loss, label='train')
ax[0].plot(val_loss, label='test')
ax[0].plot(idx_best_mode, np.min(val_loss), 'rx', label='best model\nloss=%.4f' %np.min(val_loss))
ax[0].set_xlabel('epochs'), ax[0].set_ylabel('loss')
ax[0].legend()

ax[1].plot(R2, label='train')
ax[1].plot(val_R2, label='test')
ax[1].plot(idx_best_mode, val_R2[idx_best_mode], 'rx')
ax[1].set_ylim(-0.1,1)
ax[1].set_xlabel('epochs'), ax[1].set_ylabel(r'$R^2$')

plt.savefig('%sloss_plot.png' %path, bbox_inches='tight')