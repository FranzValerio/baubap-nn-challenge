#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import train_test_split
import pandas as pd
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras


# In[2]:


tf.test.is_gpu_available()


# In[3]:


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.initializers import HeNormal, glorot_uniform
from keras.regularizers import l1, l2
from tensorflow.keras.utils import to_categorical


# In[4]:


df_train = pd.read_pickle('../../Base de datos/Processed/train_clean.pkl')
df_train


# In[5]:


df_temp = pd.read_pickle('../../Base de datos/Processed/nn_challenge_test_clean.pkl')
df_temp


# # Balanceo de datos
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
df_upsampled.target.value_counts()# Submuestreo de la clase mayoritaria
df_majority_downsampled = resample(df_majority, 
                                   replace=False,    # Muestreo sin reemplazo
                                   n_samples=len(df_minority),  # Coincide con el número en la clase minoritaria
                                   random_state=42) 

# Combinar la clase minoritaria con la clase mayoritaria submuestreada
df_downsampled = pd.concat([df_majority_downsampled, df_minority])

# Mostrar el nuevo recuento de clases
df_downsampled.target.value_counts()df_downsampled
# In[6]:


X_train = df_train.drop(columns = ['target'])
y_train = df_train['target']

y_train = to_categorical(y_train, num_classes=2)
y_train[5]
# In[7]:


X_temp = df_temp.drop(columns = ['target'])
y_temp = df_temp['target']

y_temp = to_categorical(y_temp, num_classes=2)
y_temp[5]
# # Reescalamos datos

# In[8]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)


# In[9]:


X_train


# In[10]:


X_temp = scaler.fit_transform(X_temp)


# In[11]:


X_temp


# In[12]:


from sklearn.model_selection import train_test_split

# Luego, dividimos el conjunto temporal en conjuntos de validación y prueba
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape


# In[13]:


np.shape(X_val)


# In[14]:


y_val


# # Unimos coeficeintes con datos

# In[15]:


train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))


# In[16]:


train_dataset


# # Mezclar y procesar por lotes los conjuntos de datos

# In[17]:


BATCH_SIZE = 10000
SHUFFLE_BUFFER_SIZE = 10000

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
val_dataset = val_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)


# # Creación del modelo

# In[18]:


from keras.models import Sequential
from keras.layers import Dense, Flatten, BatchNormalization, Dropout
from keras.regularizers import l1

model = Sequential()

# Flatten layer
model.add(Flatten(input_shape=(137,)))

# Batch normalization
model.add(BatchNormalization())

# First set of dense layers
model.add(Dense(120, activation='relu', kernel_initializer='glorot_uniform'))
model.add(Dense(120, activation='relu', kernel_initializer='glorot_uniform'))
model.add(Dense(120, activation='relu', kernel_initializer='glorot_uniform'))
model.add(Dropout(0.2))


# Second set of dense layers
model.add(Dense(60, activation='relu', kernel_initializer='glorot_uniform'))
model.add(Dense(60, activation='relu', kernel_initializer='glorot_uniform'))

# Dropout layer
model.add(Dropout(0.2))

# Another batch normalization
model.add(BatchNormalization())

# Output layer
model.add(Dense(1, activation='sigmoid'))




# In[19]:


model.summary()


# In[20]:


keras.utils.plot_model(model,show_shapes=True)


# # Entrenamos el modelo

# In[21]:


from keras.optimizers import Adam
optimizer = Adam(learning_rate=0.0001,clipvalue=100.0)


# In[22]:


def total_mae_loss(y_true, y_pred):
    total_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    tf.print(y_true, summarize = -1)
    tf.print(y_pred, summarize = -1)
    return total_loss


# In[23]:


def brier_score(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


# In[24]:


model.compile(optimizer= optimizer,
              loss= 'binary_crossentropy',
              metrics=[brier_score, 'accuracy'])


# In[25]:


def scheduler(epoch, lr):
    if epoch < 200:
        return 0.001
    elif epoch >= 200 and epoch <= 1000:
        slope = (0.000001 - 0.001) / (1000 - 200)
        intercept = 0.001 - (slope * 200)
        return slope * epoch + intercept
    else:
        return 0.000001  # Mantener el learning rate en 0.000001 para épocas mayores a 1000 si es necesario


# In[30]:


from keras.callbacks import Callback
import keras.backend as K

class CyclicLR(Callback):
    def __init__(self, base_lr=0.0001, max_lr=0.006, step_size=2000., mode='triangular'):
        super(CyclicLR, self).__init__()
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.mode == 'triangular':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x))

    def on_train_begin(self, logs={}):
        logs = logs or {}
        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, epoch, logs=None):
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1
        K.set_value(self.model.optimizer.lr, self.clr())
        
#    def on_epoch_end(self, epoch, logs=None):
#        # Imprimir el learning rate actual
#        current_lr = K.get_value(self.model.optimizer.lr)
#        print(f"Current Learning Rate at end of epoch {epoch}: {current_lr}")    


# In[31]:


class PrintLastBatch(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        for batch, (x, y) in enumerate(val_dataset):  # Cambia 'val_dataset' por 'train_dataset' si deseas usar datos de entrenamiento
            pass  # Este bucle se ejecutará hasta el último batch
        
        y_pred = self.model.predict(x)
        
        tf.print("Último y_true de la época:", y, summarize=-1)
        tf.print("Último y_pred de la época:", y_pred, summarize=-1)


# In[ ]:


val_epochs = 10000

early_stop = tf.keras.callbacks.EarlyStopping( monitor = 'val_accuracy', patience = 1000,verbose = 1, 
                                              restore_best_weights = True)

class_weights = {0: 1, 1: .1}  # asigna más peso a la clase 0

clr = CyclicLR(base_lr=0.0001, max_lr=0.006, step_size=2000., mode='triangular')

reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)
history = model.fit(train_dataset, validation_data=val_dataset, epochs= val_epochs, callbacks=[clr, early_stop],
                    class_weight=class_weights)
#history = model.fit(train_dataset, validation_data=val_dataset, epochs= val_epochs, callbacks=[reduce_lr, early_stop, PrintLastBatch()],
#                   class_weight=class_weights)


# # Analizamos accuracy y loss

# In[82]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

brier_score = history.history['brier_score']
val_brier_score = history.history['val_brier_score']

# Crear figura y ejes
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Graficar Accuracy
axes[0].plot(epochs_range, acc, label='Training Accuracy')
axes[0].plot(epochs_range, val_acc, label='Validation Accuracy')
axes[0].legend(loc='lower right')
axes[0].set_title('Training and Validation Accuracy')

# Graficar Loss
axes[1].plot(epochs_range, loss, label='Training Loss')
axes[1].plot(epochs_range, val_loss, label='Validation Loss')
axes[1].legend(loc='upper right')
axes[1].set_title('Training and Validation Loss')

# Graficar Brier Score
axes[2].plot(epochs_range, brier_score, label='Training Brier Score')
axes[2].plot(epochs_range, val_brier_score, label='Validation Brier Score')
axes[2].legend(loc='upper right')
axes[2].set_title('Training and Validation Brier Score')

plt.show()


# In[83]:


test_loss, test_accuracy, test_brier_score = model.evaluate(
    X_test, y_test)


# In[84]:


X_test


# In[74]:


indices_of_zero = y_test[y_test == 0.0].index
indices_of_zero


# In[75]:


y_test[131456]


# In[86]:


predictions = model.predict(X_test)


# # Guardamos datos accuracy y loss

# In[87]:


df = pd.DataFrame.from_dict(history.history)
df.to_csv('../../Modelos enternados/history2.csv', index=False)


# # Guardamos el modelo

# In[90]:


path_to_save = '../../Modelos enternados/'


# In[91]:


model.save(path_to_save + 'Modelo2.h5')

