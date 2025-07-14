"""
Image-to-Image Translation using cGAN (pix2pix architecture)
Project for ProDigy Infotech
Author: AI Assistant
Date: July 2025

This implementation demonstrates the pix2pix model for conditional image-to-image translation
using TensorFlow/Keras with a U-Net generator and PatchGAN discriminator.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, MeanAbsoluteError
import tensorflow_datasets as tfds

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Configuration parameters
class Config:
    """Configuration class to store hyperparameters"""
    IMG_WIDTH = 256
    IMG_HEIGHT = 256
    BUFFER_SIZE = 1000
    BATCH_SIZE = 1
    EPOCHS = 50  # Increased for better results like the tutorial
    LAMBDA = 100  # Weight for L1 loss
    LEARNING_RATE = 2e-4
    BETA_1 = 0.5  # Adam optimizer beta_1 parameter

    # Training checkpoint and sample directories
    CHECKPOINT_DIR = './training_checkpoints'
    SAMPLE_DIR = './samples'

config = Config()

# ==================== DATA PREPROCESSING ====================

def load_facades_dataset():
    """
    Load the facades dataset using the same approach as TensorFlow's pix2pix tutorial.
    This loads the proper paired facade dataset with concatenated images.
    """
    print("Loading facades dataset...")

    try:
        # Use the direct URL approach like TensorFlow's official tutorial
        _URL = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz'

        path_to_zip = tf.keras.utils.get_file(
            'facades.tar.gz',
            origin=_URL,
            extract=True)

        path_to_dataset = os.path.join(os.path.dirname(path_to_zip), 'facades')

        # Get train and test image paths
        train_path = os.path.join(path_to_dataset, 'train')
        test_path = os.path.join(path_to_dataset, 'test')

        # Create datasets from image files
        train_dataset = tf.data.Dataset.list_files(os.path.join(train_path, '*.jpg'))
        test_dataset = tf.data.Dataset.list_files(os.path.join(test_path, '*.jpg'))

        print(f"Found dataset at: {path_to_dataset}")
        return train_dataset, test_dataset

    except Exception as e:
        print(f"Error downloading facades dataset: {e}")
        print("Trying TensorFlow datasets approach...")

        try:
            # Try the cycle_gan approach as backup
            dataset, metadata = tfds.load('cycle_gan/facades',
                                         with_info=True,
                                         as_supervised=False)

            # Extract trainA and testA
            train_dataset = dataset['trainA']
            test_dataset = dataset['testA']

            # The cycle_gan dataset has separate A and B domains, we need to create pairs
            # For facades: A = segmentation maps, B = real photos
            train_b_dataset = dataset['trainB']
            test_b_dataset = dataset['testB']

            # Create paired dataset by zipping A and B
            train_paired = tf.data.Dataset.zip((train_dataset, train_b_dataset))
            test_paired = tf.data.Dataset.zip((test_dataset, test_b_dataset))

            return train_paired, test_paired

        except Exception as e2:
            print(f"Error loading cycle_gan/facades: {e2}")
            print("Creating synthetic facade-like dataset...")
            return create_facade_like_dataset()

def load_custom_dataset(data_path):
    """
    Load custom paired image dataset from a directory.
    Expected structure: data_path/train/, data_path/test/
    Each image should be 512x256 (input|target concatenated horizontally)
    """
    def process_path(file_path):
        image = tf.io.read_file(file_path)
        image = tf.image.decode_image(image, channels=3)
        image = tf.cast(image, tf.float32)
        return image

    train_path = os.path.join(data_path, 'train', '*.jpg')
    test_path = os.path.join(data_path, 'test', '*.jpg')

    train_dataset = tf.data.Dataset.list_files(train_path)
    train_dataset = train_dataset.map(process_path)

    test_dataset = tf.data.Dataset.list_files(test_path)
    test_dataset = test_dataset.map(process_path)

    return train_dataset, test_dataset

def create_facade_like_dataset():
    """
    Create a synthetic dataset that mimics the facade segmentation to real building translation.
    This creates colorful segmentation-like patterns and corresponding realistic textures.
    """
    print("Creating facade-like synthetic dataset...")

    def generate_facade_pair():
        """Generate a synthetic facade segmentation and corresponding realistic image."""
        # Create segmentation-like input with distinct colored regions
        h, w = 256, 256

        # Create base colors for different building parts
        colors = [
            [255, 0, 0],      # Red - walls
            [0, 255, 0],      # Green - doors
            [0, 0, 255],      # Blue - windows
            [255, 255, 0],    # Yellow - roof
            [255, 0, 255],    # Magenta - decorative elements
            [0, 255, 255],    # Cyan - structural elements
        ]

        # Create segmentation map
        input_img = np.zeros((h, w, 3), dtype=np.float32)

        # Add rectangular regions with different colors (simulating building segments)
        regions = [
            (0, 0, 256, 60, colors[3]),      # Roof
            (0, 60, 256, 180, colors[0]),    # Main wall
            (20, 80, 60, 140, colors[2]),    # Windows
            (80, 80, 120, 140, colors[2]),   # Windows
            (140, 80, 180, 140, colors[2]),  # Windows
            (196, 80, 236, 140, colors[2]),  # Windows
            (100, 180, 156, 220, colors[1]), # Door
            (0, 220, 256, 256, colors[5]),   # Foundation
        ]

        for x1, y1, x2, y2, color in regions:
            input_img[y1:y2, x1:x2] = color

        # Add some noise to make it more realistic
        noise = np.random.uniform(-20, 20, (h, w, 3))
        input_img = np.clip(input_img + noise, 0, 255)

        # Create realistic-looking target image
        target_img = np.zeros((h, w, 3), dtype=np.float32)

        # Convert segmentation to realistic textures
        for y in range(h):
            for x in range(w):
                # Sample the input color
                input_color = input_img[y, x]

                # Map to realistic building textures
                if np.allclose(input_color, colors[3], atol=30):  # Roof -> dark gray
                    target_img[y, x] = [60, 60, 60] + np.random.normal(0, 10, 3)
                elif np.allclose(input_color, colors[0], atol=30):  # Wall -> beige/brown
                    target_img[y, x] = [180, 150, 120] + np.random.normal(0, 15, 3)
                elif np.allclose(input_color, colors[2], atol=30):  # Windows -> dark blue
                    target_img[y, x] = [40, 60, 80] + np.random.normal(0, 5, 3)
                elif np.allclose(input_color, colors[1], atol=30):  # Door -> brown
                    target_img[y, x] = [80, 40, 20] + np.random.normal(0, 10, 3)
                elif np.allclose(input_color, colors[5], atol=30):  # Foundation -> gray
                    target_img[y, x] = [100, 100, 100] + np.random.normal(0, 10, 3)
                else:  # Default to wall color
                    target_img[y, x] = [180, 150, 120] + np.random.normal(0, 15, 3)

        # Ensure values are in valid range
        target_img = np.clip(target_img, 0, 255)

        return tf.constant(input_img), tf.constant(target_img)

    # Create synthetic datasets
    train_size = 1000
    test_size = 100

    train_dataset = tf.data.Dataset.from_generator(
        lambda: (generate_facade_pair() for _ in range(train_size)),
        output_signature=(
            tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32)
        )
    )

    test_dataset = tf.data.Dataset.from_generator(
        lambda: (generate_facade_pair() for _ in range(test_size)),
        output_signature=(
            tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32)
        )
    )

    return train_dataset, test_dataset

def load_image_file(image_file):
    """
    Load and preprocess a single image file (for the direct facade dataset).
    Expected format: concatenated image with input|target side by side.
    """
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)

    # Split the image into input and target (assuming 512x256 -> 256x256 each)
    w = tf.shape(image)[1]
    w = w // 2

    input_image = image[:, :w, :]
    target_image = image[:, w:, :]

    # Convert to float32 and normalize
    input_image = tf.cast(input_image, tf.float32)
    target_image = tf.cast(target_image, tf.float32)

    return input_image, target_image

def preprocess_paired_images(input_image, target_image):
    """
    Preprocess paired images from the dataset.

    Args:
        input_image: Input image tensor
        target_image: Target image tensor

    Returns:
        Preprocessed input and target images
    """
    # Handle different input formats
    if isinstance(input_image, dict):
        input_image = input_image['image']
    if isinstance(target_image, dict):
        target_image = target_image['image']

    # Resize images to target size
    input_image = tf.image.resize(input_image, [config.IMG_HEIGHT, config.IMG_WIDTH])
    target_image = tf.image.resize(target_image, [config.IMG_HEIGHT, config.IMG_WIDTH])

    # Normalize to [-1, 1] range
    input_image = tf.cast(input_image, tf.float32)
    target_image = tf.cast(target_image, tf.float32)

    input_image = (input_image / 127.5) - 1
    target_image = (target_image / 127.5) - 1

    return input_image, target_image

def random_jitter(input_image, target_image):
    """
    Apply random jittering for data augmentation during training.

    Args:
        input_image: Source image
        target_image: Target image

    Returns:
        Augmented input and target images
    """
    # Resize to larger size
    input_image = tf.image.resize(input_image, [286, 286])
    target_image = tf.image.resize(target_image, [286, 286])

    # Random crop back to original size
    stacked_image = tf.stack([input_image, target_image], axis=0)
    cropped_image = tf.image.random_crop(stacked_image,
                                        size=[2, config.IMG_HEIGHT, config.IMG_WIDTH, 3])

    input_image, target_image = cropped_image[0], cropped_image[1]

    # Random horizontal flip
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        target_image = tf.image.flip_left_right(target_image)

    return input_image, target_image

def create_dataset(raw_dataset, is_training=True):
    """
    Create processed dataset pipeline.

    Args:
        raw_dataset: Raw dataset from load function
        is_training: Whether this is training dataset (applies augmentation)

    Returns:
        Processed dataset ready for training/testing
    """
    # Check if this is a file path dataset (from direct download) or paired dataset
    sample = next(iter(raw_dataset.take(1)))

    if isinstance(sample, tf.Tensor):
        # This is a file path dataset - process each file
        dataset = raw_dataset.map(load_image_file, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        # This is already a paired dataset
        dataset = raw_dataset.map(preprocess_paired_images, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        # Apply data augmentation for training
        dataset = dataset.map(random_jitter, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(config.BUFFER_SIZE)

    # Batch and prefetch for performance
    dataset = dataset.batch(config.BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

# ==================== GENERATOR (U-Net) ====================

def downsample(filters, size, apply_batchnorm=True):
    """
    Downsampling block for U-Net encoder.

    Args:
        filters: Number of filters
        size: Kernel size
        apply_batchnorm: Whether to apply batch normalization

    Returns:
        Sequential model for downsampling
    """
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(layers.Conv2D(filters, size, strides=2, padding='same',
                            kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(layers.BatchNormalization())

    result.add(layers.LeakyReLU())

    return result

def upsample(filters, size, apply_dropout=False):
    """
    Upsampling block for U-Net decoder.

    Args:
        filters: Number of filters
        size: Kernel size
        apply_dropout: Whether to apply dropout

    Returns:
        Sequential model for upsampling
    """
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                     kernel_initializer=initializer, use_bias=False))
    result.add(layers.BatchNormalization())

    if apply_dropout:
        result.add(layers.Dropout(0.5))

    result.add(layers.ReLU())

    return result

def Generator():
    """
    U-Net based Generator for pix2pix.

    Architecture:
    - Encoder: Series of downsampling blocks
    - Decoder: Series of upsampling blocks with skip connections

    Returns:
        Generator model
    """
    inputs = layers.Input(shape=[config.IMG_HEIGHT, config.IMG_WIDTH, 3])

    # Encoder (Downsampling)
    down_stack = [
        downsample(64, 4, apply_batchnorm=False),   # (bs, 128, 128, 64)
        downsample(128, 4),                         # (bs, 64, 64, 128)
        downsample(256, 4),                         # (bs, 32, 32, 256)
        downsample(512, 4),                         # (bs, 16, 16, 512)
        downsample(512, 4),                         # (bs, 8, 8, 512)
        downsample(512, 4),                         # (bs, 4, 4, 512)
        downsample(512, 4),                         # (bs, 2, 2, 512)
        downsample(512, 4),                         # (bs, 1, 1, 512)
    ]

    # Decoder (Upsampling)
    up_stack = [
        upsample(512, 4, apply_dropout=True),       # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),       # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),       # (bs, 8, 8, 1024)
        upsample(512, 4),                           # (bs, 16, 16, 1024)
        upsample(256, 4),                           # (bs, 32, 32, 512)
        upsample(128, 4),                           # (bs, 64, 64, 256)
        upsample(64, 4),                            # (bs, 128, 128, 128)
    ]

    # Final layer
    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(3, 4, strides=2, padding='same',
                                 kernel_initializer=initializer,
                                 activation='tanh')  # (bs, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate()([x, skip])

    x = last(x)

    return Model(inputs=inputs, outputs=x)

# ==================== DISCRIMINATOR (PatchGAN) ====================

def Discriminator():
    """
    PatchGAN Discriminator for pix2pix.

    The discriminator classifies 70x70 patches of the image as real or fake.
    This approach focuses on local image structure.

    Returns:
        Discriminator model
    """
    initializer = tf.random_normal_initializer(0., 0.02)

    # Input image
    inp = layers.Input(shape=[config.IMG_HEIGHT, config.IMG_WIDTH, 3], name='input_image')
    # Target image
    tar = layers.Input(shape=[config.IMG_HEIGHT, config.IMG_WIDTH, 3], name='target_image')

    # Concatenate input and target images
    x = layers.concatenate([inp, tar])  # (bs, 256, 256, 6)

    # Downsampling layers
    down1 = downsample(64, 4, False)(x)     # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1)       # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2)       # (bs, 32, 32, 256)

    # Zero padding
    zero_pad1 = layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)

    # Convolution layer
    conv = layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer,
                        use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

    batchnorm1 = layers.BatchNormalization()(conv)
    leaky_relu = layers.LeakyReLU()(batchnorm1)

    # Zero padding
    zero_pad2 = layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

    # Final convolution layer
    last = layers.Conv2D(1, 4, strides=1,
                        kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

    return Model(inputs=[inp, tar], outputs=last)

# ==================== LOSS FUNCTIONS ====================

def discriminator_loss(disc_real_output, disc_generated_output):
    """
    Calculate discriminator loss.

    Args:
        disc_real_output: Discriminator output for real images
        disc_generated_output: Discriminator output for generated images

    Returns:
        Total discriminator loss
    """
    loss_object = BinaryCrossentropy(from_logits=True)

    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss

def generator_loss(disc_generated_output, gen_output, target):
    """
    Calculate generator loss (adversarial + L1 loss).

    Args:
        disc_generated_output: Discriminator output for generated images
        gen_output: Generated images
        target: Target images

    Returns:
        Total generator loss, adversarial loss, L1 loss
    """
    loss_object = BinaryCrossentropy(from_logits=True)

    # Adversarial loss
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # L1 loss
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    # Total generator loss
    total_gen_loss = gan_loss + (config.LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss

# ==================== TRAINING LOOP ====================

class Pix2Pix:
    """
    Main Pix2Pix training class.
    """

    def __init__(self):
        """Initialize the Pix2Pix model with generator and discriminator."""
        self.generator = Generator()
        self.discriminator = Discriminator()

        # Optimizers
        self.generator_optimizer = Adam(config.LEARNING_RATE, beta_1=config.BETA_1)
        self.discriminator_optimizer = Adam(config.LEARNING_RATE, beta_1=config.BETA_1)

        # Checkpoints
        self.checkpoint_prefix = os.path.join(config.CHECKPOINT_DIR, "ckpt")
        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.generator,
            discriminator=self.discriminator
        )

        # Create directories
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(config.SAMPLE_DIR, exist_ok=True)

        # Training metrics
        self.train_loss = {'gen_total': [], 'gen_gan': [], 'gen_l1': [], 'disc': []}

    @tf.function
    def train_step(self, input_image, target, step):
        """
        Execute one training step.

        Args:
            input_image: Input image batch
            target: Target image batch
            step: Current training step

        Returns:
            Generator and discriminator losses
        """
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate fake image
            gen_output = self.generator(input_image, training=True)

            # Discriminator predictions
            disc_real_output = self.discriminator([input_image, target], training=True)
            disc_generated_output = self.discriminator([input_image, gen_output], training=True)

            # Calculate losses
            gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(
                disc_generated_output, gen_output, target)
            disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

        # Calculate gradients
        generator_gradients = gen_tape.gradient(gen_total_loss, self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        # Apply gradients
        self.generator_optimizer.apply_gradients(
            zip(generator_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_variables))

        return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss

    def generate_images(self, test_input, tar, step, save_path=None):
        """
        Generate and display/save sample images during training.

        Args:
            test_input: Test input image
            tar: Target image
            step: Current training step
            save_path: Path to save the generated image
        """
        prediction = self.generator(test_input, training=False)

        plt.figure(figsize=(15, 5))

        display_list = [test_input[0], tar[0], prediction[0]]
        title = ['Input Segmentation', 'Ground Truth Facade', 'Generated Facade']

        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.title(title[i], fontsize=14, fontweight='bold')
            # Convert from [-1, 1] to [0, 1] for display
            img_display = display_list[i] * 0.5 + 0.5
            plt.imshow(img_display)
            plt.axis('off')

        plt.suptitle(f'Pix2Pix Facade Translation - Step {step}', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)

        plt.show()

        # Also show individual images for better comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Input segmentation with color legend
        axes[0].imshow(test_input[0] * 0.5 + 0.5)
        axes[0].set_title('Input: Segmentation Map\n(Colors represent building parts)', fontsize=12)
        axes[0].axis('off')

        # Ground truth
        axes[1].imshow(tar[0] * 0.5 + 0.5)
        axes[1].set_title('Ground Truth: Real Facade', fontsize=12)
        axes[1].axis('off')

        # Generated
        axes[2].imshow(prediction[0] * 0.5 + 0.5)
        axes[2].set_title('Generated: Predicted Facade', fontsize=12)
        axes[2].axis('off')

        plt.tight_layout()
        if save_path:
            detailed_path = save_path.replace('.png', '_detailed.png')
            plt.savefig(detailed_path, bbox_inches='tight', dpi=150)

        plt.show()

    def fit(self, train_ds, test_ds, steps_per_epoch):
        """
        Train the pix2pix model.

        Args:
            train_ds: Training dataset
            test_ds: Test dataset
            steps_per_epoch: Number of steps per epoch
        """
        # Get a sample from test dataset for visualization
        try:
            example_input, example_target = next(iter(test_ds.take(1)))
        except:
            print("Warning: Could not get test sample for visualization")
            example_input, example_target = None, None

        print(f"Starting training for {config.EPOCHS} epochs...")
        print(f"Steps per epoch: {steps_per_epoch}")

        start_time = time.time()

        for epoch in range(config.EPOCHS):
            epoch_start = time.time()

            # Training loop
            step = 0
            for input_image, target in train_ds:
                if step >= steps_per_epoch:
                    break

                gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss = self.train_step(
                    input_image, target, step)

                # Print progress every 10 steps
                if step % 10 == 0:
                    print(f"Epoch {epoch+1}/{config.EPOCHS}, Step {step+1}/{steps_per_epoch}, "
                          f"Gen Loss: {gen_total_loss:.4f}, Disc Loss: {disc_loss:.4f}")

                step += 1

            # Record losses (use the last batch's losses)
            self.train_loss['gen_total'].append(gen_total_loss.numpy())
            self.train_loss['gen_gan'].append(gen_gan_loss.numpy())
            self.train_loss['gen_l1'].append(gen_l1_loss.numpy())
            self.train_loss['disc'].append(disc_loss.numpy())

            # Generate sample images every 5 epochs
            if (epoch + 1) % 5 == 0 and example_input is not None:
                save_path = os.path.join(config.SAMPLE_DIR, f'epoch_{epoch+1}.png')
                self.generate_images(example_input, example_target, epoch+1, save_path)

            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)
                print(f"Saved checkpoint at epoch {epoch+1}")

            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds")

        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds")

        # Plot training losses
        self.plot_training_losses()

    def plot_training_losses(self):
        """Plot training losses over epochs."""
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.plot(self.train_loss['gen_total'], label='Generator Total Loss')
        plt.title('Generator Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(self.train_loss['gen_gan'], label='Generator GAN Loss')
        plt.plot(self.train_loss['gen_l1'], label='Generator L1 Loss')
        plt.title('Generator Loss Components')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(self.train_loss['disc'], label='Discriminator Loss')
        plt.title('Discriminator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.plot(self.train_loss['gen_total'], label='Generator Total')
        plt.plot(self.train_loss['disc'], label='Discriminator')
        plt.title('Generator vs Discriminator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(config.SAMPLE_DIR, 'training_losses.png'))
        plt.show()

# ==================== MAIN EXECUTION ====================

def main():
    """
    Main function to run the pix2pix training pipeline.
    """
    print("=== Pix2Pix Image-to-Image Translation ===")
    print("ProDigy Infotech Project")
    print("=" * 50)

    # Check GPU availability
    print(f"GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

    try:
        # Load dataset
        print("\n1. Loading dataset...")
        train_raw, test_raw = load_facades_dataset()

        # Create datasets
        print("2. Creating training and test datasets...")
        train_dataset = create_dataset(train_raw, is_training=True)
        test_dataset = create_dataset(test_raw, is_training=False)

        # Calculate steps per epoch
        try:
            # Try to get dataset size for better progress tracking
            steps_per_epoch = 100  # Default fallback
            sample_count = 0
            for _ in train_dataset.take(100):
                sample_count += 1
            if sample_count > 0:
                steps_per_epoch = sample_count
        except:
            steps_per_epoch = 100

        print(f"Estimated steps per epoch: {steps_per_epoch}")

        # Initialize model
        print("\n3. Initializing Pix2Pix model...")
        pix2pix = Pix2Pix()

        # Print model summaries
        print("\n4. Model architectures:")
        print("Generator Summary:")
        pix2pix.generator.summary()

        print("\nDiscriminator Summary:")
        pix2pix.discriminator.summary()

        # Start training
        print("\n5. Starting training...")
        pix2pix.fit(train_dataset, test_dataset, steps_per_epoch)

        print("\n6. Training completed successfully!")
        print("=" * 50)
        print("RESULTS SUMMARY:")
        print(f"• Total training time: {(time.time() - start_time):.1f} seconds")
        print(f"• Final Generator Loss: {pix2pix.train_loss['gen_total'][-1]:.4f}")
        print(f"• Final Discriminator Loss: {pix2pix.train_loss['disc'][-1]:.4f}")
        print(f"• Generated samples saved in: {config.SAMPLE_DIR}")
        print(f"• Model checkpoints saved in: {config.CHECKPOINT_DIR}")
        print("\nYour pix2pix model is now trained and ready for facade translation!")
        print("The model should translate segmentation maps to realistic building facades,")
        print("similar to the results shown in the TensorFlow tutorial.")

        # Generate a few final test images
        print("\n7. Generating final test results...")
        test_sample = next(iter(test_dataset.take(1)))
        for i in range(min(3, len(test_sample[0]))):
            test_input = tf.expand_dims(test_sample[0][i], 0)
            test_target = tf.expand_dims(test_sample[1][i], 0)
            final_path = os.path.join(config.SAMPLE_DIR, f'final_result_{i+1}.png')
            pix2pix.generate_images(test_input, test_target, f"Final Test {i+1}", final_path)

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("Make sure you have internet connection to download the dataset.")
        print("Alternatively, you can use custom dataset by calling load_custom_dataset()")

def test_model_on_custom_images(model_path, input_images):
    """
    Test trained model on custom images.

    Args:
        model_path: Path to saved model checkpoint
        input_images: List of input image paths
    """
    # Load trained model
    pix2pix = Pix2Pix()
    pix2pix.checkpoint.restore(model_path)

    for img_path in input_images:
        # Load and preprocess image
        image = tf.io.read_file(img_path)
        image = tf.image.decode_image(image, channels=3)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, [config.IMG_HEIGHT, config.IMG_WIDTH])
        image = (image / 127.5) - 1  # Normalize to [-1, 1]
        image = tf.expand_dims(image, axis=0)  # Add batch dimension

        # Generate prediction
        prediction = pix2pix.generator(image, training=False)

        # Display result
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title('Input Image')
        plt.imshow(image[0] * 0.5 + 0.5)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title('Generated Image')
        plt.imshow(prediction[0] * 0.5 + 0.5)
        plt.axis('off')

        plt.show()

if __name__ == "__main__":
    main()

# ==================== USAGE EXAMPLES ====================

"""
USAGE EXAMPLES:

1. Basic training with facades dataset:
   python pix2pix.py

2. Using custom dataset:
   train_raw, test_raw = load_custom_dataset('/path/to/your/dataset')
   # Then follow the same training pipeline

3. Testing on custom images after training:
   test_model_on_custom_images('/path/to/checkpoint', ['image1.jpg', 'image2.jpg'])

4. Adjusting hyperparameters:
   config.EPOCHS = 200
   config.BATCH_SIZE = 2
   config.LAMBDA = 50
   # Then run training

5. Loading pretrained model:
   pix2pix = Pix2Pix()
   pix2pix.checkpoint.restore('/path/to/checkpoint')
   # Now you can use pix2pix.generator for inference
"""
