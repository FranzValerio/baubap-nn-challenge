#!/usr/bin/env python
# coding: utf-8

# In[127]:


from sklearn.model_selection import train_test_split
import pandas as pd
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras


# In[128]:


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.initializers import HeNormal
from keras.regularizers import l1, l2
from tensorflow.keras.utils import to_categorical


# In[129]:


df = pd.read_pickle('../../Base de datos/train_clean.pkl')
df


# # Balanceo de datos

# In[130]:


from sklearn.utils import resample

# Separar las clases mayoritaria y minoritaria
df_majority = df[df.target==1]
df_minority = df[df.target==0]

# Sobremuestreo de la clase minoritaria
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # Muestreo con reemplazo
                                 n_samples=len(df_majority),    # Coincide con el número en la clase mayoritaria
                                 random_state=42) 

# Combinar la clase mayoritaria con la clase minoritaria sobremuestreada
df_upsampled = pd.concat([df_majority, df_minority_upsampled])

# Mostrar el nuevo recuento de clases
df_upsampled.target.value_counts()


# In[131]:


# Submuestreo de la clase mayoritaria
df_majority_downsampled = resample(df_majority, 
                                   replace=False,    # Muestreo sin reemplazo
                                   n_samples=len(df_minority),  # Coincide con el número en la clase minoritaria
                                   random_state=42) 

# Combinar la clase minoritaria con la clase mayoritaria submuestreada
df_downsampled = pd.concat([df_majority_downsampled, df_minority])

# Mostrar el nuevo recuento de clases
df_downsampled.target.value_counts()


# In[132]:


df_downsampled


# In[133]:


X = df_downsampled.drop(columns = ['target'])
y = df_downsampled['target']


# In[134]:


y


# In[135]:


X


# In[136]:


y = to_categorical(y, num_classes=2)
y


# # Reescalamos datos

# In[137]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[138]:


X_scaled


# In[139]:


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.3, random_state=42)


# In[140]:


np.shape(X_val)


# # Unimos coeficeintes con datos

# In[141]:


train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))


# In[142]:


train_dataset


# # Mezclar y procesar por lotes los conjuntos de datos

# In[143]:


BATCH_SIZE = 100
SHUFFLE_BUFFER_SIZE = 100

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
val_dataset = val_dataset.batch(BATCH_SIZE)


# # Creación del modelo

# In[158]:


from keras.models import Sequential
from keras.layers import Dense, Flatten, BatchNormalization, Dropout
from keras.regularizers import l1

model = Sequential()

# Flatten layer
model.add(Flatten(input_shape=(120,)))

# Batch normalization
model.add(BatchNormalization())

# First set of dense layers
model.add(Dense(120, activation='relu', kernel_initializer='glorot_uniform'))
model.add(Dense(120, activation='relu', kernel_initializer='glorot_uniform'))
model.add(Dense(120, activation='relu', kernel_initializer='glorot_uniform'))

# Second set of dense layers
model.add(Dense(60, activation='relu', kernel_initializer='glorot_uniform'))
model.add(Dense(60, activation='relu', kernel_initializer='glorot_uniform'))

# Dropout layer
model.add(Dropout(0.2))

# Another batch normalization
model.add(BatchNormalization())

# Output layer
model.add(Dense(2, activation='sigmoid'))



# In[159]:


model.summary()


# In[160]:


keras.utils.plot_model(model,show_shapes=True)


# # Entrenamos el modelo

# In[161]:


from keras.optimizers import Adam
optimizer = Adam(learning_rate=0.001,clipvalue=100.0)


# In[162]:


def total_mae_loss(y_true, y_pred):
    total_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    tf.print(y_true, summarize = -1)
    tf.print(y_pred, summarize = -1)
    return total_loss


# In[163]:


def brier_score(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

metrics = [ ]

model.compile(optimizer='adam',
              loss= 'binary_crossentropy',
              metrics=['accuracy'])
# In[164]:


model.compile(optimizer= optimizer,
              loss= 'binary_crossentropy',
              metrics=[brier_score, 'accuracy'])


# In[165]:


def scheduler(epoch, lr):
  if epoch < 400:
    return lr
  else:
    return -1.65e-6*epoch +  0.00166


# In[166]:


class PrintLastBatch(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        for batch, (x, y) in enumerate(val_dataset):  # Cambia 'val_dataset' por 'train_dataset' si deseas usar datos de entrenamiento
            pass  # Este bucle se ejecutará hasta el último batch
        
        y_pred = self.model.predict(x)
        
        tf.print("Último y_true de la época:", y, summarize=-1)
        tf.print("Último y_pred de la época:", y_pred, summarize=-1)


# In[175]:


val_epochs = 1000

early_stop = tf.keras.callbacks.EarlyStopping( monitor = 'val_accuracy', patience = 1000,verbose = 1, 
                                              restore_best_weights = True)

#class_weights = {0: 1, 1: .16}  # asigna más peso a la clase 0


reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)
#history = model.fit(train_dataset, validation_data=val_dataset, epochs= val_epochs, callbacks=[reduce_lr, early_stop],
#                    class_weight=class_weights)
history = model.fit(train_dataset, validation_data=val_dataset, epochs= val_epochs, callbacks=[reduce_lr, early_stop])


# # Analizamos accuracy y loss

# In[168]:


# Plotting
plt.figure(figsize=(18, 6))

# Subplot for Accuracy
plt.subplot(1, 3, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# Subplot for Loss
plt.subplot(1, 3, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

# Subplot for Brier Score
plt.subplot(1, 3, 3)
plt.plot(epochs_range, brier_score, label='Training Brier Score')
plt.plot(epochs_range, val_brier_score, label='Validation Brier Score')
plt.legend(loc='upper right')
plt.title('Training and Validation Brier Score')

plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[169]:


evaluación = X_val[100]


# In[ ]:





# In[ ]:





# In[ ]:





# In[170]:


X_test1_None = evaluación[None, :]


# In[171]:


prediction = model.predict(X_test1_None)
print(prediction)


# In[172]:


y_val[10]


# In[173]:


df.dtypes.value_counts()


# In[ ]:




