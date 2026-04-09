import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ---------------- PARAMETERS ----------------
IMG_SIZE = 64
LATENT_DIM = 100
EPOCHS = 2000
SAVE_DIR = "dataset/train/1. Eczema 1677"

# ---------------- LOAD REAL IMAGES ----------------
def load_images(folder):
    images = []
    for file in os.listdir(folder):
        img = load_img(os.path.join(folder, file), target_size=(IMG_SIZE, IMG_SIZE))
        img = img_to_array(img)
        img = (img - 127.5) / 127.5   # Normalize to [-1, 1]
        images.append(img)
    return np.array(images)

real_images = load_images(SAVE_DIR)

# ---------------- GENERATOR ----------------
def build_generator():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8*8*128, input_dim=LATENT_DIM),
        tf.keras.layers.Reshape((8,8,128)),

        tf.keras.layers.UpSampling2D(),   # 16x16
        tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),

        tf.keras.layers.UpSampling2D(),   # 32x32
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),

        tf.keras.layers.UpSampling2D(),   # 64x64 ✅
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),

        tf.keras.layers.Conv2D(3, 3, padding='same', activation='tanh')
    ])
    return model


# ---------------- DISCRIMINATOR ----------------
def build_discriminator():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, 3, strides=2, padding='same', input_shape=(64,64,3)),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

generator = build_generator()
discriminator = build_discriminator()

discriminator.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

discriminator.trainable = False

gan_input = tf.keras.layers.Input(shape=(LATENT_DIM,))
gan_output = discriminator(generator(gan_input))
gan = tf.keras.Model(gan_input, gan_output)

gan.compile(optimizer='adam', loss='binary_crossentropy')

# ---------------- TRAIN GAN ----------------
for epoch in range(EPOCHS):
    idx = np.random.randint(0, real_images.shape[0], 32)
    real = real_images[idx]

    noise = np.random.normal(0, 1, (32, LATENT_DIM))
    fake = generator.predict(noise, verbose=0)

    d_loss_real = discriminator.train_on_batch(real, np.ones((32,1)))
    d_loss_fake = discriminator.train_on_batch(fake, np.zeros((32,1)))

    noise = np.random.normal(0, 1, (32, LATENT_DIM))
    g_loss = gan.train_on_batch(noise, np.ones((32,1)))

    if epoch % 500 == 0:
        print(f"Epoch {epoch} | Generator Loss: {g_loss}")

# ---------------- GENERATE SYNTHETIC IMAGES ----------------
noise = np.random.normal(0, 1, (50, LATENT_DIM))
generated_images = generator.predict(noise)

generated_images = (generated_images + 1) * 127.5

for i, img in enumerate(generated_images):
    tf.keras.preprocessing.image.save_img(
        f"{SAVE_DIR}/synthetic_{i}.jpg", img
    )

print("✅ Synthetic images generated successfully")