# Variational Autoencoder (VAE) for MNIST Digit Generation
## Introduction
Welcome to my project on a Python-based implementation of a Variational Autoencoder (VAE), designed to generate new, handwritten digits using the MNIST dataset. VAEs stand at the forefront of generative models, capable of learning deep representations of data and then generating new instances that resemble the original dataset. By modeling the underlying probability distribution of data, VAEs allow us to explore the complex manifold that represents handwritten digits in a way that's both mathematically rigorous and practically fascinating.

The MNIST dataset, a cornerstone in the machine learning community, consists of thousands of handwritten digits, making it an ideal playground for our VAE. My project not only demonstrates the power of VAEs but also serves as an educational tool for those interested in delving into the world of AI/ML, particularly in the realm of image generation. 
Note: This project was done under Prof. Pantelis Monogioudis as part of my CSGY-6613: Introduction to Artificial Intelligence course at NYU. 


## Project Overview
This project employs TensorFlow for constructing the VAE model and Optuna for hyperparameter optimization, ensuring the model's performance is maximized. Through detailed setup instructions, users can replicate our results or explore the model's capabilities further.

**Key Features:**

**TensorFlow Integration:** Leveraging TensorFlow for efficient model construction and training.
**Optuna for Hyperparameter Tuning:** Utilizing Optuna to find the optimal latent space dimension for our VAE.
**Visual Demonstrations:** Including generated digit images, latent space visualizations, and loss curves during training to illustrate the learning process.


## Getting Started
**Prerequisites**
Ensure you have Python 3.6+ installed on your system. Additionally, you will need the following libraries:

TensorFlow (GPU version is preferred but the generic/CPU version will also do)
Optuna
NumPy
Matplotlib

You can install these with the following command:
```bash
pip install tensorflow optuna numpy matplotlib
```

## Installation and Usage
1. Clone this repository to your local machine.
2. Navigate to the project directory.
3. Run the VAE_MNIST.py script to start the training and optimization process:
```bash
python VAE_MNIST.py
```
4. Experiment with different model parameters or a whole new dataset by modifying the script as needed.


## Project Structure
VAE_MNIST.py: Main script containing the VAE model, training, and optimization logic.
output/vae_generated_images: Directory containing images generated by the VAE.
output/latent_space_visualizations: Visualizations of the latent space, showing the distribution and separation of digits.
output/loss_curves: Training and validation loss curves.


## Future Directions
I envision several exciting paths for further developing this project, including:

**Exploring Additional Datasets:** Applying the VAE model to other image datasets.
**Refining Model Architecture:** Enhancing the model's structure for improved performance.
**Advanced VAE Variants:** Integrating more sophisticated VAE variants into the project.


## Contributing
I welcome contributions from the community, whether in the form of code enhancements, documentation improvements, or issue reporting. Together, we can push the boundaries of what's possible with generative models in AI/ML.


## License
This project is open-sourced under the MIT license. Feel free to use, modify, and distribute it as you see fit.
