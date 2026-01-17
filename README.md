# Wireless Interference Detection using Deep Learning

This project presents an end-to-end system for detecting and classifying wireless
interference using deep learning applied to raw baseband I/Q signals. The system
combines classical digital communication signal modeling with modern convolutional
neural networks to identify abnormal and interfering signal conditions.

The project is designed to reflect realistic wireless communication scenarios
and focuses on interpretability, reproducibility, and robust evaluation.

---

## Problem Description

Wireless communication systems are frequently affected by noise and interference,
which can significantly degrade performance. Automatic detection and
classification of interference is a key task in modern communication systems,
including cellular networks, satellite communications, and IoT deployments.

This project addresses the problem by learning discriminative features directly
from raw I/Q samples, without relying on handcrafted signal features.

---

## Signal Model

All signals are generated in complex baseband representation (I/Q). The following
signal conditions are considered:

- **Class 0:** Clean QPSK signal
- **Class 1:** QPSK signal with additive white Gaussian noise (AWGN)
- **Class 2:** QPSK signal with narrowband sinusoidal interference
- **Class 3:** QPSK signal with impulsive interference

Signals are generated synthetically to allow full control over modulation,
noise, and interference characteristics.

---

## Dataset

- Modulation: QPSK
- Representation: I/Q baseband samples
- Samples per signal: 1024
- Channels: 2 (I and Q)
- Total dataset size: 2000 signals (balanced across classes)

The dataset is generated automatically and labeled according to the interference
type.

---

## Methodology

### Data Generation
- Random QPSK symbol sequences
- AWGN added at fixed SNR
- Narrowband interference modeled as a sinusoidal tone
- Impulsive interference modeled as sparse high-amplitude noise bursts

### Model Architecture
- 1D Convolutional Neural Network (CNN)
- Input: (2 × 1024) I/Q samples
- Three convolutional blocks with ReLU activations and max pooling
- Fully connected classifier head

### Training
- Loss function: Cross-Entropy Loss
- Optimizer: Adam
- Train/validation split: 80/20
- Supervised multi-class classification

---

## Experimental Results

### Classification Performance

The trained model achieves high performance on the validation set:

- **Overall accuracy:** ~97%
- Perfect classification of clean QPSK, AWGN, and narrowband interference
- Most errors occur between impulsive interference and AWGN

Confusion matrix analysis shows that impulsive interference is occasionally
misclassified as noise, which is expected due to their similar stochastic
behavior in short observation windows.

---

### Signal Visualization

Time-domain and frequency-domain visualizations were generated for each signal
class. These plots confirm that:

- Clean QPSK signals exhibit discrete amplitude levels in time
- Narrowband interference produces distinct spectral peaks
- Impulsive interference introduces broadband spectral components

These visualizations reinforce the interpretability of the learned model.

---

## Conclusions

This project demonstrates that deep learning models can effectively detect and
classify different types of wireless interference directly from raw I/Q signals.

The proposed CNN achieves high classification accuracy across multiple interference
conditions while maintaining strong interpretability through signal-domain
visualization. Observed misclassifications align with known signal processing
behavior, highlighting the physical consistency of the learned representations.

Overall, the results show that combining classical communication theory with deep
learning provides a practical and powerful framework for interference detection
in modern wireless systems.

---

## Repository Structure

wireless-interference-detection/
│
├── data/
│ ├── signal_generator.py
│ └── dataset.py
│
├── src/
│ ├── model.py
│ ├── train.py
│ ├── evaluate.py
│ └── visualize.py
│
├── results/
│
├── notebooks/
│
├── requirements.txt
└── README.md


---

## Reproducibility

To reproduce the full pipeline:

```bash
pip install -r requirements.txt
python data/signal_generator.py
python -m src.train
python -m src.evaluate
python -m src.visualize

Technologies Used

Python

PyTorch

NumPy

SciPy

Signal Processing

Digital Communications

Git & GitHub

Author

Carlos Navarro
