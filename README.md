# Text Normalization for Indic Languages

This repository presents an ensemble-based approach for text normalization in Indic languages. Text normalization remains a relatively underexplored problem for many Indic languages, with one of the primary challenges being the lack of high-quality training data. To address this, we propose a synthetic data generation pipeline that leverages Large Language Models (LLMs) and the Kestrel dataset to create text normalization datasets for target Indic languages.

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

The `data_generation` folder contains the complete pipeline for generating synthetic text normalization datasets from the Kestrel dataset. Follow the instructions provided within the folder to create datasets for new Indic languages from scratch.

### Model Weights and Training Data

The script `import_models_&_training_data.py` can be used to download:

* Pretrained model weights for Hints_ib model
* Training datasets
* Dynamic model router weights

for Hindi and Kannada.

### LLM and Seq2Seq Models

The `LLM_and_seq2seq` folder contains training and inference code for:

* Gemma
* Llama
* mT5
* IndicBART

Additional details are provided within the folder.

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
To have the compatible MT5 for ensemble use the data obtained from **imporrt_models_&_training_data.py** and train the mt5 small on that using code given in  folder **LLM and Seq2Seq models**

through a dynamic model selection framework.

## Dataset Requirement

The synthetic data generation pipeline requires access to the Kestrel dataset. Using the provided pipeline, datasets can be generated for additional Indic languages by leveraging machine translation and LLM-based normalization.
