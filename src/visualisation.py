import pickle as pkl
import keras
from matplotlib import axes
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# ## Open History files of each model.
# ## Model 1.
# path = os.path.join('src', 'saves', 'a5_c_1', 'history', 'history.pkl')
# file = open(path, 'rb')
# hist1 = pkl.load(file)
# file.close()
# ## Model 3.
# path = os.path.join('src', 'saves', 'a5_c_2', 'history', 'history.pkl')
# file = open(path, 'rb')
# hist2 = pkl.load(file)
# file.close()
# ## Model 3.
# path = os.path.join('src', 'saves', 'a4_3', 'history', 'history.pkl')
# file = open(path, 'rb')
# hist3 = pkl.load(file)
# file.close()
# ## Model 4.
# path = os.path.join('src', 'saves', 'a3_4', 'history', 'history.pkl')
# file = open(path, 'rb')
# hist4 = pkl.load(file)
# file.close()



# ## Plot Training Data
# # Plot Loss Function.
# plt.figure()
# plt.xlabel('Epochs')
# plt.ylabel('Cross Entropy')
# plt.plot(hist1['loss'], label='128 neurons')
# plt.plot(hist2['loss'], label='256 neurons')
# plt.plot(hist3['loss'][:250], label='coeff = .9')
# plt.plot(hist4['loss'][:250], label='lr=.1 | m=.6')
# plt.legend(loc='best')
# # Plot MSE.
# plt.figure()
# plt.xlabel('Epochs')
# plt.ylabel('Mean Square Error')
# plt.plot(hist1['mse'], label='128 neurons')
# plt.plot(hist2['mse'], label='256 neurons')
# plt.plot(hist3['mse'][:250], label='coeff = .9')
# plt.plot(hist4['mse'][:250], label='lr=.1 | m=.6')
# plt.legend(loc='best')
# # Plot Accuracy.
# plt.figure()
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.plot(hist1['accuracy'], label='128 neurons')
# plt.plot(hist2['accuracy'], label='256 neurons')
# plt.plot(hist3['accuracy'][:250], label='coeff = .9')
# plt.plot(hist4['accuracy'][:250], label='lr=.1 | m=.6')
# plt.legend(loc='best')
# plt.show()

# # Plot Training Data
# # Plot Loss Function.
# plt.figure()
# plt.xlabel('Epochs')
# plt.ylabel('Cross Entropy')
# plt.plot(hist1['loss'], label='2135 Neurons')
# plt.plot(hist2['loss'], label='4270 Neurons')
# plt.plot(hist3['loss'][:250], label='8540 Neurons')
# # plt.plot(hist4['loss'][:250], label='lr=.1 | m=.6')
# plt.legend(loc='best')
# # Plot MSE.
# plt.figure()
# plt.xlabel('Epochs')
# plt.ylabel('Mean Square Error')
# plt.plot(hist1['mse'], label='coeff = .1')
# plt.plot(hist2['mse'], label='coeff = .5')
# plt.plot(hist3['mse'][:250], label='coeff = .9')
# # plt.plot(hist4['mse'][:250], label='lr=.1 | m=.6')
# plt.legend(loc='best')
# # Plot Accuracy.
# plt.figure()
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.plot(hist1['accuracy'], label='coeff = .1')
# plt.plot(hist2['accuracy'], label='coeff = .5')
# plt.plot(hist3['accuracy'][:250], label='coeff = .9')
# # plt.plot(hist4['accuracy'][:250], label='lr=.1 | m=.6')
# plt.legend(loc='best')
# plt.show()


## Open History files of each model.
# Model 1.
path = os.path.join('src', 'saves', 'a5_c_1', 'evaluation', 'eval.pkl')
file = open(path, 'rb')
hist1 = pkl.load(file)
file.close()
# Model 3.
path = os.path.join('src', 'saves', 'a5_c_2', 'evaluation', 'eval.pkl')
file = open(path, 'rb')
hist2 = pkl.load(file)
file.close()
# # Model 3.
# path = os.path.join('src', 'saves', 'a4_3', 'evaluation', 'eval.pkl')
# file = open(path, 'rb')
# hist3 = pkl.load(file)
# file.close()
# # Model 4.
# path = os.path.join('src', 'saves', 'a3_4', 'evaluation', 'eval.pkl')
# file = open(path, 'rb')
# hist4 = pkl.load(file)
# file.close()

print(hist1)
print(hist2)
# print(hist3)
# print(hist4)
