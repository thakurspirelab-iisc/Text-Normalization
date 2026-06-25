## Indic_BERT_with_Hints

This directory contains the training and inference code for both components of the **Hints_IB** model:

- **NER Model**: An IndicBERT-based token classification model used to identify and classify unnormalized spans in the input sentence.
- **Decoder Model**: A decoder-only model derived from IndicBERT that generates normalized spoken-form representations for the detected spans using rule-based hints.

### Training Data

The input dataset follows the same format described in the repository root README. Each line of the training file is a dictionary (or JSON object) containing:

- `tagged_ip` (or `unnorm` / `unnormalized`) — the tagged unnormalized input sentence.
- `normalized_op` — the corresponding tagged normalized output sentence.

During preprocessing, tags are removed while preserving the enclosed text. The cleaned sentences are then used for training the NER and decoder models.
just specify the dataset file path and path for model_weights in each training file
