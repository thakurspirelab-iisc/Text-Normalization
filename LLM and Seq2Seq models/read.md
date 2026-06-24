The following models were fine-tuned using supervised fine-tuning (SFT) on the generated dataset:

* **Gemma 1B**, **Llama 1B**, and **mT5 Large (≈1B parameters)** were fine-tuned using **LoRA (Low-Rank Adaptation)**.
* **Gemma 270M** was trained using a combination of **LoRA-based fine-tuning** and **knowledge distillation**, with **Gemma 1B** serving as the teacher model.
* **IndicBART**,**MT5 small**  was fully fine-tuned by updating all model parameters.

This setup enables a comparison between full fine-tuning, parameter-efficient fine-tuning (LoRA), and knowledge-distillation-based approaches for text normalization in Indic languages.



