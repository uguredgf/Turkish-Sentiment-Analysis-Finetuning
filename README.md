# 🚀 Turkish Sentiment Analysis | CodeNight Hackathon

This repository contains my solution for the Turkish Sentiment Analysis competition held during the **Fırat University FÜB3V CodeNight** event. The competition ran overnight from 22:00 to 05:00, requiring rapid prototyping, efficient resource management, and high-performance NLP modeling under severe time constraints.

## 🏆 Results
* **Rank:** 6th Place
* **Accuracy:** 83.86% (Top score: ~86%)

## 🧠 Tech Stack & Model Architecture
* **Framework:** Native PyTorch (Custom training loop)
* **Model:** `xlm-roberta-large` (State-of-the-art multilingual transformer)
* **Libraries:** Transformers (Hugging Face), Pandas, NumPy, Emoji

## ⚡ Key Engineering Decisions & "The 10-Minute Doping"
Due to strict time limits and GPU memory constraints (VRAM) during the hackathon, several critical engineering choices were made rather than relying on standard wrappers:

1. **Hardware Optimization:** Implemented strict max_length constraints (`max_len=64`), optimal batch sizing, and limited the training to only 2 epochs to fit the massive XLM-RoBERTa-Large model into limited VRAM without triggering OOM (Out of Memory) errors.
2. **Context-Aware Preprocessing:** Instead of discarding emojis using standard regex (which destroys sentiment value), I utilized the `emoji` library to convert them into text representations (e.g., ` EMOJI `), preserving crucial sentiment features for the model.
3. **Pseudo-Labeling (The Doping Strategy):** In the final 10 minutes of the competition, I implemented a pseudo-labeling technique. High-confidence predictions from the test set were reinjected into the training loop, significantly boosting the model's accuracy right before the submission deadline!
4. **Custom PyTorch Loop:** Avoided high-level wrappers like Hugging Face `Trainer` to maintain granular control over gradient clipping (`torch.nn.utils.clip_grad_norm_`), learning rate scheduling, and backpropagation.

## 🚀 How to Run

1. Clone the repository:
```bash
git clone [https://github.com/uguredgf/Turkish-Sentiment-Analysis-Finetuning.git](https://github.com/uguredgf/Turkish-Sentiment-Analysis-Finetuning.git)
cd Turkish-Sentiment-Analysis-Finetuning
```
2. Install the required dependencies:
```bash
pip install torch transformers pandas numpy emoji accelerate
```

3. Run the Jupyter Notebook:
Open `turkish-sentiment-analysis-xlm-roberta-doping.ipynb` and run the cells. The script will automatically detect whether it is running on Kaggle or a local machine and adjust the data paths accordingly.
