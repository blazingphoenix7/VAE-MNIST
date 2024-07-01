import optuna
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Ensure output directories exist
os.makedirs("output", exist_ok=True)
os.makedirs("output/vae_generated_images", exist_ok=True)

# Use Agg backend for matplotlib
plt.switch_backend('Agg')

def build_and_train_vae(latent_space_dim, input_dim=28*28, hidden_layer_size=64, batch_size_value=100, training_epochs=50, epsilon_standard=1.0):
    """
    Build and train a Variational Autoencoder.

    Args:
        latent_space_dim (int): Dimension of the latent space.
        input_dim (int): Dimension of the original input.
        hidden_layer_size (int): Size of hidden layers.
        batch_size_value (int): Batch size for training.
        training_epochs (int): Number of epochs for training.
        epsilon_standard (float): Standard deviation for epsilon.

    Returns:
        Tuple: Contains training history, trained encoder, generator, test data.
    """

    # Encoder setup
    input_layer = Input(shape=(input_dim,))
    hidden_layer = Dense(hidden_layer_size, activation='relu')(input_layer)
    latent_mean = Dense(latent_space_dim)(hidden_layer)
    latent_log_variance = Dense(latent_space_dim)(hidden_layer)

    # Reparameterization trick
    def reparametrize(args):
        mean, log_variance = args
        epsilon = K.random_normal(shape=(K.shape(mean)[0], latent_space_dim), mean=0., stddev=epsilon_standard)
        return mean + K.exp(log_variance / 2) * epsilon

    z_layer = Lambda(reparametrize, output_shape=(latent_space_dim,))([latent_mean, latent_log_variance])

    # Decoder setup
    decoder_hidden = Dense(hidden_layer_size, activation='relu')
    decoder_output_layer = Dense(input_dim, activation='sigmoid')
    output_layer = decoder_output_layer(decoder_hidden(z_layer))

    # VAE model construction
    vae_model = Model(input_layer, output_layer)

    # Loss function computation
    reconstruction_loss = input_dim * tf.keras.losses.binary_crossentropy(input_layer, output_layer)
    kl_divergence_loss = -0.5 * K.sum(1 + latent_log_variance - K.square(latent_mean) - K.exp(latent_log_variance), axis=-1)
    total_vae_loss = K.mean(reconstruction_loss + kl_divergence_loss)

    vae_model.add_loss(total_vae_loss)
    vae_model.compile(optimizer='rmsprop')

    # Data loading and preprocessing
    (train_data, _), (test_data, test_labels) = mnist.load_data()
    train_data = train_data.reshape((-1, input_dim)).astype('float32') / 255
    test_data = test_data.reshape((-1, input_dim)).astype('float32') / 255

    # VAE model training
    train_history = vae_model.fit(train_data, shuffle=True, epochs=training_epochs, batch_size=batch_size_value, validation_data=(test_data, None))

    # Separating encoder and generator models
    encoder_model = Model(input_layer, latent_mean)
    decoder_input = Input(shape=(latent_space_dim,))
    generated_output = decoder_output_layer(decoder_hidden(decoder_input))
    generator_model = Model(decoder_input, generated_output)

    return train_history, encoder_model, generator_model, test_data, test_labels

def optuna_objective(trial):
    """
    Objective function for Optuna optimization.

    Args:
        trial (optuna.Trial): Optuna trial instance.

    Returns:
        float: The validation loss of the trained model.
    """

    # Latent dimension optimization
    latent_dimension = trial.suggest_int('latent_dimension', 1, 20)

    # Training the VAE with given latent dimension
    history, encoder, generator, x_test, y_test = build_and_train_vae(latent_dimension)

    # Plotting and saving training and validation losses
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Training and Validation Loss (Latent Dim: {latent_dimension})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"output/loss_curve_trial_{trial.number}_latent_dim_{latent_dimension}.png")
    plt.close()

    # Plotting and saving latent space visualization
    encoded_test = encoder.predict(x_test)
    plt.figure(figsize=(6, 6))
    plt.scatter(encoded_test[:, 0], encoded_test[:, 1] if latent_dimension > 1 else np.zeros_like(encoded_test), c=y_test, cmap='viridis')
    plt.colorbar()
    plt.title(f'Latent Space (Latent Dim: {latent_dimension})')
    plt.xlabel('z_1')
    plt.ylabel('z_2' if latent_dimension > 1 else '')
    plt.savefig(f"output/latent_space_trial_{trial.number}_latent_dim_{latent_dimension}.png")
    plt.close()

    # Generating and saving images for digits 0-9
    plt.figure(figsize=(20, 4))
    for i in range(10):
        sample_z = np.random.normal(size=(1, latent_dimension))
        x_decoded = generator.predict(sample_z)
        digit_image = x_decoded[0].reshape(28, 28)
        plt.subplot(1, 10, i + 1)
        plt.imshow(digit_image, cmap='gray')
        plt.axis('off')
    plt.savefig(f"output/vae_generated_images/vae_digits_trial_{trial.number}_latent_dim_{latent_dimension}.png")
    plt.close()

    return history.history['val_loss'][-1]

# Initiate and run Optuna optimization
optuna_study = optuna.create_study(direction='minimize')
optuna_study.optimize(optuna_objective, n_trials=5)

# Reporting best latent dimension size
print("Best Latent Dimension Size:", optuna_study.best_params['latent_dimension']) 
