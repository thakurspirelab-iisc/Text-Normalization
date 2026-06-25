# Text Normalization for Indic Languages

This repository presents an ensemble-based approach for text normalization in Indic languages. Text normalization remains a relatively underexplored problem for many Indic languages, largely due to the scarcity of high-quality annotated training data. To address this challenge, we propose a synthetic data generation pipeline that leverages Large Language Models (LLMs) and the Kestrel dataset to create text normalization datasets for target Indic languages.

## Ensemble Architecture

The proposed ensemble system combines the strengths of multiple normalization approaches:

* **Rule-Based Model (RB)**
* **mT5 Small**
* **Hints_IB (IndicBERT with Hints)**

### Hints_IB (IndicBERT with Hints)

Hints_IB consists of two stages:

1. An **IndicBERT-based NER model** identifies and tags unnormalized spans in the input sentence.
2. The detected spans are passed to a modified **IndicBERT decoder**, along with rule-based hints, to generate their normalized spoken-form representations.

The normalized spans are then substituted back into the original sentence to produce the final normalized output.

## Repository Structure

### Data Generation

The `data_generation` folder contains the complete pipeline for generating synthetic text normalization datasets from the Kestrel dataset. Detailed instructions are provided within the folder for generating datasets in new Indic languages from scratch.

### Model Weights and Training Data

The script `import_models_&_training_data.py` can be used to download:

* Pretrained Hints_IB model weights
* Training datasets
* Dynamic model router weights

for both Hindi and Kannada.

### LLM and Seq2Seq Models

The `LLM_and_seq2seq` folder contains training and inference code for:

* Gemma
* Llama
* mT5
* IndicBART

Additional details and usage instructions are available within the folder.

### Indic_BERT_with_Hints

The `Indic_BERT_with_Hints` folder contains the training and inference code for the encoder and decoder components of the Hints_IB model.

### Rule_Based_Models

The `Rule_Based_Models` folder contains rule-based text normalization systems for:

* Hindi
* Kannada

### Ensemble_Models

The `Ensemble_Models` folder contains the training and inference code for the final ensemble system, which combines:

* Rule-Based normalization
* mT5 Small
* Hints_IB

through a dynamic model selection framework.

**Important:** To obtain an mT5 model compatible with the ensemble framework, first download the training data using `import_models_&_training_data.py`. The downloaded dataset can then be used to train mT5 Small using the training scripts provided in the `LLM_and_seq2seq` folder.

## Dataset Requirement

The synthetic data generation pipeline requires access to the Kestrel dataset. Using the provided pipeline, datasets can be generated for additional Indic languages through machine translation and LLM-based normalization, enabling the creation of training resources for low-resource text normalization tasks.
