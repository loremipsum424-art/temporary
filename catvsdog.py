# 1. import libraries 

import os 
import tensorflow as tf 
import matplotlib.pyplot as plt 
from tensorflow import keras 
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau


# 2. paths 

TRAIN_DIR= './catvsdog/train'
VAL_DIR= './catvsdog/test'
OUT_DIR= './output'

os.makedirs(OUT_DIR,exist_ok=True)

if not os.path.exists(TRAIN_DIR) or not os.path.exists(VAL_DIR):
    raise FileNotFoundError(
        f"Couldnt find directories.\n"
        f"Train: {os.path.abspath(TRAIN_DIR)}\n"
        f"Test/Val: {os.path.abspath(VAL_DIR)}\n"
    )

print("Dataset folders found! Loading images.....")


# 3. data loading

train_ds = keras.utils.image_dataset_from_directory(
    directory= TRAIN_DIR,
    labels= 'inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256,256),
    shuffle=False
)



validation_ds=keras.utils.image_dataset_from_directory(
    directory=VAL_DIR,
     labels='inferred',
     label_mode='int',
     batch_size=32,
     image_size=(256,256),
     shuffle=  False
)


# 4. preprocessing

def process(image,label):
    image=tf.cast(image,tf.float32)/255.0
    return image,label

train_ds = train_ds.map(process).cache().prefetch(tf.data.AUTOTUNE)
validation_ds = validation_ds.map(process).cache().prefetch(tf.data.AUTOTUNE)


# 5. build model 

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), padding='valid',activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D(pool_size=(2,2),strides=2, padding='valid'))

model.add(Conv2D(64, kernel_size=(3,3), padding='valid',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2, padding='valid'))

model.add(Conv2D(128, kernel_size=(3,3), padding='valid',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2, padding='valid'))

model.add(Flatten())

model.add(Dense(128,activation='relu'))

model.add(Dense(64,activation='relu'))

model.add(Dense(1,activation='sigmoid'))





# 6. compile model 

model.compile (optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])




#summary of the model 

model.summary()



# 7. callbacks and training

early_stopping= EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

history= model.fit(
  train_ds,
  epochs= 100,
  validation_data=validation_ds,
  callbacks=[early_stopping, reduce_lr],
  verbose= 1
 )


model.save(os.path.join(OUT_DIR, 'cat_dog_cnn.keras'))
print(f"model saved successfully inside {OUT_DIR}!")





# 8. plot graphs 

acc= history.history.get('accuracy',[])
val_acc= history.history.get('val_accuracy',[])
loss= history.history.get('loss',[])
val_loss= history.history.get('val_loss',[])

epochs_range=range(1,len(acc)+1)

plt.figure(fig_size=(12,5))

plt.subplot(1,2,1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.title('Accuracy Evaluation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)



plt.subplot(1,2,2)
plt.plot(epochs_range, loss, label='Training loss')
plt.plot(epochs_range, val_loss, label='Vaidation loss')
plt.title('Loss Evaluation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)



plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,'training_curve.png'),dpi=200)
plt.show()





