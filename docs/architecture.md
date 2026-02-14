# System Architecture

## Overview

This repository implements an intelligent multimodal monitoring system
for microalgae culture growth based on computer vision, environmental
sensing, and neural network inference deployed on edge devices.

The system follows a hybrid architecture combining:

-   Data acquisition (embedded hardware)
-   Image and signal processing
-   Machine learning training pipeline
-   Edge inference deployment
-   Data storage and analysis

The design prioritizes:

-   Modular reproducibility
-   Low-cost hardware integration
-   Edge AI execution
-   Scientific experimentation traceability

------------------------------------------------------------------------

## High-Level Architecture

The system is composed of five main layers:

    ┌──────────────────────────────┐
    │        Experiment Layer        │
    │  Photobioreactor + Sensors     │
    └───────────────┬───────────────┘
                    │
    ┌───────────────▼───────────────┐
    │     Data Acquisition Layer      │
    │        ESP32 Firmware           │
    └───────────────┬───────────────┘
                    │
    ┌───────────────▼───────────────┐
    │      Processing Layer           │
    │ Image preprocessing + Fusion    │
    └───────────────┬───────────────┘
                    │
    ┌───────────────▼───────────────┐
    │      Machine Learning Layer     │
    │ Training / Validation Pipeline  │
    └───────────────┬───────────────┘
                    │
    ┌───────────────▼───────────────┐
    │        Edge Inference Layer     │
    │ Quantized Neural Network        │
    │ deployed on ESP32               │
    └───────────────────────────────┘

------------------------------------------------------------------------

## Component Description

### 1. Experiment Layer

Physical system responsible for biological data generation.

**Components** - Tubular photobioreactor - RGB camera - Temperature
sensor - pH sensor - Illumination subsystem - Aeration subsystem

**Output** - RGB images - Environmental variables

------------------------------------------------------------------------

### 2. Data Acquisition Layer

Implemented on an ESP32 microcontroller.

**Responsibilities** - Sensor reading synchronization - Environmental
monitoring - Control signals - Data transmission

**Characteristics** - Real-time operation - Low-power execution -
Deterministic sampling cycle

------------------------------------------------------------------------

### 3. Processing Layer

Executed on a workstation/server.

**Pipeline**

1.  Image capture
2.  Region of Interest (ROI) extraction
3.  Noise reduction
4.  RGB statistical feature extraction
5.  Environmental signal normalization
6.  Multimodal feature vector construction

**Output**

    Feature Vector (10D):
    [R_mean, G_mean, B_mean, I_mean,
    R_std, G_std, B_std, I_std,
    Temperature, pH]

------------------------------------------------------------------------

### 4. Machine Learning Layer

Responsible for model development.

**Model** - Multilayer Perceptron (MLP)

**Workflow** - Dataset partitioning - Training - Cross-validation -
Performance evaluation - Robustness testing

**Metrics** - Accuracy - Precision - Recall - F1-score

------------------------------------------------------------------------

### 5. Edge Inference Layer

Deployment of optimized neural network on embedded hardware.

**Process** 1. Model export (Keras → TensorFlow Lite) 2. INT8
quantization 3. Conversion to C byte array 4. Firmware integration

**Execution** - Local inference on ESP32 - Real-time growth phase
classification

------------------------------------------------------------------------

## Data Flow

    Sensors + Camera
            ↓
    ESP32 Acquisition
            ↓
    Server Processing
            ↓
    Feature Extraction
            ↓
    Model Training
            ↓
    Quantization
            ↓
    Embedded Deployment
            ↓
    Real-time Inference

------------------------------------------------------------------------

## Repository Structure

    project-root/
    │
    ├── firmware/           # ESP32 code
    ├── data/
    │   ├── raw/
    │   ├── processed/
    │   └── datasets/
    │
    ├── ml/
    │   ├── training/
    │   ├── evaluation/
    │   └── models/
    │
    ├── image_processing/
    │
    ├── deployment/
    │   └── tflite_export/
    │
    ├── docs/
    │   └── architecture/
    │
    └── notebooks/

------------------------------------------------------------------------

## Design Principles

-   Modular separation between hardware and ML
-   Experiment reproducibility
-   Edge-first inference
-   Hardware-aware optimization
-   Scientific traceability

------------------------------------------------------------------------

## Future Extensions

-   IoT connectivity
-   Cloud synchronization
-   Automated environmental control
-   Multi-species biological models
