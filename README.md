

# Modular Neural Inference for Biological Growth Monitoring

AI-driven framework for real-time analysis and prediction of microalgae growth using RGB image processing and deep learning.

MSc Thesis Project — Computer Science

<img src="./logo.png" alt="Logo de bioharvest" width="100%"/>

## Abstract
Monitoring microalgae cultures remains a significant challenge in biotechnology, as accurate identification of physiological growth stages typically relies on invasive techniques, subjective manual assessments, or high-cost commercial systems, limiting their adoption in small- and medium-scale research and production environments. These limitations highlight the need for automated, non-invasive, and computationally efficient solutions capable of enabling continuous real-time culture monitoring.

This work proposes an integrated intelligent framework for cellular growth monitoring in _Chlorella vulgaris_ cultures based on multimodal data fusion combining computer vision and environmental sensing. A vertically oriented tubular photobioreactor instrumented with low-cost sensors was designed and constructed to acquire synchronized RGB images and physicochemical variables, including pH and temperature. Over a 98-day experimental period, a dataset comprising 2,352 samples was collected, from which morphochromatic and environmental features were extracted to form a multidimensional feature vector. A Multilayer Perceptron (MLP) neural network was trained to automatically classify microalgal growth phases.

The proposed model achieved an average accuracy of 98.62% and a macro F1-score of 98.93%, demonstrating strong discriminative capability and robustness under operational perturbations such as image noise and blur. The model was subsequently optimized through INT8 quantization and deployed on an ESP32 microcontroller following an edge-computing approach, enabling efficient real-time inference with reduced computational and memory requirements.

The main contribution of this research lies in demonstrating the feasibility of integrating multimodal artificial intelligence and embedded systems for autonomous, non-invasive monitoring of bioprocesses, providing an accessible and scalable alternative to traditional industrial solutions and establishing a foundation for future intelligent control systems in microalgae cultivation.

## Research Motivation

## Contributions

## System Architecture

## Repository Structure

## Installation

## Dataset

## Training

## Experiments

## Paper

## Future Work

## License
