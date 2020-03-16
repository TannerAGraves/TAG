import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
# In[3]:
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train_full,y_train_full), (X_test,y_test) = fashion_mnist.load_data()
# In[4]:
X_valid = X_train_full[:5000] / 255.0
X_train = X_train_full[5000:] / 255.0
X_test = X_test / 255.0
y_valid = y_train_full[:5000]
y_train = y_train_full[5000:]
X_train = X_train[..., np.newaxis]
X_valid = X_valid[..., np.newaxis]
X_test = X_test[..., np.newaxis]
# In[5]:
from functools import partial
my_dense_layer = partial(tf.keras.layers.Dense, activation="tanh",kernel_regularizer=tf.keras.regularizers.l2(0.0001))
my_conv_layer = partial(tf.keras.layers.Conv2D, activation="tanh",padding="valid")
example = tf.keras.models.Sequential([
    my_conv_layer(6,5,padding="same",input_shape=[28,28,1]),
    tf.keras.layers.AveragePooling2D(2),
    my_conv_layer(16,5),
    tf.keras.layers.AveragePooling2D(2),
    my_conv_layer(120,5),
    tf.keras.layers.Flatten(),
    my_dense_layer(84),
    my_dense_layer(10, activation="softmax")
])
# In[38]:
from functools import partial
my_dense_layer = partial(tf.keras.layers.Dense, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(0.0001))
my_conv_layer = partial(tf.keras.layers.Conv2D, activation="relu",padding="same")
myModel = tf.keras.models.Sequential([
    my_conv_layer(8,3,input_shape=[28,28,1]),
    my_conv_layer(16,5),
    tf.keras.layers.MaxPooling2D(2),
    my_conv_layer(16,3),
    my_conv_layer(32,14),
    tf.keras.layers.MaxPooling2D(2),
    my_conv_layer(32,3),
    my_conv_layer(32,7,padding="valid"),
    tf.keras.layers.Flatten(),
    my_dense_layer(84),
    my_dense_layer(10, activation="softmax")
])
myModel.summary()
# In[39]:
myModel.compile(loss="sparse_categorical_crossentropy",
             optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
             #optimizer = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9),
             metrics=["accuracy"])
# In[57]:
history = big.fit(X_train,y_train, epochs=5,validation_data=(X_valid,y_valid))
# In[24]:
from sklearn.model_selection import KFold
##K-fold trainer
mdl = example
dataX = X_train_full
dataY = y_train_full
dataX = dataX[..., np.newaxis]
n_folds = 5
scores, histories = list(),list()
kfold = KFold(n_folds, shuffle=True, random_state=1)
for train_ix,valid_ix in kfold.split(dataX):
    mdl.compile(loss="sparse_categorical_crossentropy",
             #optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
             optimizer = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9),
             metrics=["accuracy"])
    trainX, trainY, validX, validY = dataX[train_ix], dataY[train_ix], dataX[valid_ix], dataY[valid_ix]
    history = mdl.fit(trainX,trainY,epochs=10, batch_size=32,validation_data=(validX,validY))
    _,acc = mdl.evaluate(validX,validY)
    scores.append(acc)
    histories.append(history)
# In[42]:
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()
# In[52]:
y_pred = myModel.predict_classes(X_train)
conf_train = confusion_matrix(y_train,y_pred)
print(conf_train)
# In[44]:
myModel.evaluate(X_test,y_test)
# In[46]:
y_pred1 = myModel.predict_classes(X_test)
conf_test = confusion_matrix(y_test,y_pred1)
print(conf_test)
# In[51]:
fig,ax=plt.subplots()
fig.patch.set_visible(False)
ax.axis("off")
ax.axis("tight")
df = pd.DataFrame(conf_test)
ax.table(cellText=df.values, rowLabels=np.arange(10),colLabels=np.arange(10),loc="center",cellLoc="center")
fig.tight_layout()
#plt.savefig("conf.mat.pdf")
# In[249]:
myModel.layers[0].name
filters, biases = model.layers[0].get_weights()
# normalize filter values to 0-1 so we can visualize them
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)
# plot first few filters
n_filters = filters.shape[3]
ix = 1;
for i in range(n_filters):
    # get the filter
    f = filters[:, :, :, i]
    # plot each channel separately
    for j in range(1):
        # specify subplot and turn of axis
        ax = plt.subplot(n_filters, 30, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        # plot filter channel in grayscale
        plt.imshow(filters[:, :,j,i], cmap='gray')
        ix += 1
# show the figure
plt.show()
# In[53]:
mdl = example
garbo = tf.keras.Model(inputs=mdl.inputs, outputs=mdl.layers[0].output)
garbo.summary()
# load the image with the required shape
img = X_test[1510,:,:,:]
# expand dimensions so that it represents a single 'sample'
#img = expand_dims(img, axis=0)
img = img[np.newaxis,...]
# prepare the image (e.g. scale pixel values for the vgg)
#img = preprocess_input(img)
# get feature map for first hidden layer
feature_maps = garbo.predict(img)
# plot all 64 maps in an 8x8 squares
nrow = 2
ncol = 3
ix = 1
for _ in range(nrow):
	for _ in range(ncol):
		# specify subplot and turn of axis
		ax = plt.subplot(nrow, ncol, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		# plot filter channel in grayscale
		plt.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
		ix += 1
# show the figure
plt.show()
plt.imshow(img[0, :, :, 0],cmap="gray")
# In[31]:
from numpy import mean
from numpy import std
print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
# box and whisker plots of results
plt.boxplot(scores)
plt.title("Boxplot of example CNN: folds=5")
plt.ylabel("accuracy")
plt.show()
# In[33]:
exampleScores=scores
# In[7]:
from tensorflow.keras.preprocessing.image import ImageDataGenerator
gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                               height_shift_range=0.08, zoom_range=0.08)
batches = gen.flow(X_train, y_train, batch_size=256)
val_batches = gen.flow(X_valid, y_valid, batch_size=256)
gen.fit(X_train)
itr = gen.flow(X_train)
# In[9]:
history3 = example.fit(batches,steps_per_epoch=55000//256, epochs=50,
                    validation_data=val_batches, validation_steps=5000//256)
# In[37]:
smplBatch = batches.__getitem__(210)[0]
smplBatch.shape
ix = 1
strt = 48
for imNum in range(strt,strt+8):
    ax=plt.subplot(2, 4, ix)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.imshow(smplBatch[imNum,:,:,0],cmap="gray")
    ix = ix+1
plt.suptitle("Augmented images")
# In[39]:
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    my_dense_layer(500),
    my_dense_layer(500),
    my_dense_layer(500),
    my_dense_layer(10, activation="softmax")
])