# LLM and Seq2Seq Models

This folder contains the training and inference code for the sequence-to-sequence and Large Language Models (LLMs) used for text normalization in Indic languages.

## Supported Models

### Large Language Models (LLMs)

* **Gemma 270M**
* **Gemma 1B**
* **Llama 1B**

### Sequence-to-Sequence Models

* **mT5 Small**
* **mT5 Large (~1B parameters)**
* **IndicBART**

---

## Training

The models are trained using the synthetic text normalization datasets generated using the pipeline provided in the `data_generation` folder.

### Training Data Format

Each line of the training file is a dictionary (or JSON object) containing a tagged unnormalized sentence and its corresponding tagged normalized output.

Example:

```python
{
    "tagged_ip": "मेरी उम्र <CARDINAL>25</CARDINAL> साल है",
    "normalized_op": "मेरी उम्र <CARDINAL>पच्चीस</CARDINAL> साल है"
}
```

During training, the dataset loader removes the tags and uses:

**Input**

```text
मेरी उम्र 25 साल है
```

**Target**

```text
मेरी उम्र पच्चीस साल है
```

for model training.

---

## Fine-Tuning Strategy

### LoRA Fine-Tuning

The following models are fine-tuned using **LoRA (Low-Rank Adaptation)**:

* Gemma 1B
* Llama 1B
* mT5 Large (~1B parameters)

LoRA significantly reduces the number of trainable parameters while maintaining strong performance.

### Knowledge Distillation

**Gemma 270M** is trained using:

* LoRA-based fine-tuning
* Knowledge Distillation

with **Gemma 1B** serving as the teacher model.

### Full Fine-Tuning

The following models are fully fine-tuned by updating all model parameters:

* mT5 Small
* IndicBART

---

## Training Scripts

This folder contains separate training scripts for each model. Model-specific hyperparameters, checkpoints, and dataset paths can be configured directly within the corresponding training scripts.

---

## Inference

Inference scripts are provided for all supported models. Given an unnormalized sentence, the models generate the corresponding normalized spoken-form sentence.

Example:

**Input**

```text
मेरी उम्र 25 साल है
```

**Output**

```text
मेरी उम्र पच्चीस साल है
```

---

## Ensemble Compatibility

The ensemble framework included in the `Ensemble_Models` folder uses:

* Rule-Based Normalization
* Hints_IB
* mT5 Small

For compatibility with the ensemble system, the provided training data downloaded through `import_models_&_training_data.py` should be used to train the mT5 Small model.

The resulting mT5 Small checkpoint can then be directly integrated into the ensemble pipeline.

---

## Folder Contents

This folder includes:

* Training scripts for Gemma, Llama, mT5, and IndicBART
* Inference scripts for all models
* LoRA fine-tuning implementations
* Knowledge distillation code for Gemma 270M
* Utility scripts for data loading and preprocessing
* Model evaluation scripts

Refer to the individual model files for model-specific training and inference instructions.
