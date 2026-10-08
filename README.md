# SHL Hiring Assessment 2026 — Spoken Grammar Scoring

A multi-modal machine learning solution for predicting the grammar quality of spoken English from audio recordings.

**Latest Kaggle Score: 0.3437**

---

## Overview

I built this project as part of the **SHL Hiring Assessment 2026** challenge.

The objective is to build a Grammar Scoring Engine that takes a spoken English audio recording and predicts a continuous grammar score between **0 and 5**.

The dataset contains:

- **769 training audio samples**
- **216 evaluation audio samples**
- `.wav` speech recordings
- A continuous grammar-score target for the training set

Instead of relying on one model or one type of feature, I use a combination of **speech representations, transcript representations, explicit grammar features, acoustic features, and Whisper transcription-quality features**.

This allows the model to capture both how the speaker sounds and what the speaker says.

---

## My Result

| Metric | Result |
|---|---:|
| Latest Kaggle Score | **0.3437** |
| Training Samples | 769 |
| Test Samples | 216 |
| Prediction Range | 0–5 |

The **0.3437** score is my latest submitted Kaggle result.

---

## Approach

My pipeline has four main stages:

```text
                 Spoken Audio
                      │
          ┌───────────┴───────────┐
          │                       │
      Speech Models           Whisper ASR
          │                       │
   ┌──────┼──────┐          ┌─────┼─────┐
   │      │      │          │           │
 WavLM  Whisper  Acoustic  Transcript  Quality
 Large   Encoder Features   │           │
   │      │      │          │           │
   └──────┴──────┘          ├───────────┤
          │                 │           │
          │              MPNet      DeBERTa
          │                 │           │
          │             Grammar / Text Features
          │                 │
          └──────────┬──────┘
                     │
             Ridge / ExtraTrees
                     │
             Residual Corrections
                     │
                  LightGBM
                     │
              Final Ensemble
                     │
              Grammar Score 0–5
```

---

## 1. Speech Representations

I use pretrained speech models to extract information directly from the audio.

### WavLM-Large

I use WavLM-Large as one of the primary speech representations.

For each recording, I extract a pooled representation using the hidden states and combine mean and standard-deviation statistics.

**Representation size: 2,048 features**

### WavLM-Base

I also use WavLM-Base as a complementary speech representation.

**Representation size: 1,536 features**

### Whisper-medium Encoder

In addition to WavLM, I extract a representation from the Whisper-medium encoder.

This provides a different representation of the speech signal because Whisper is optimized for speech recognition rather than the same self-supervised objective used by WavLM.

**Representation size: 2,048 features**

---

## 2. Automatic Speech Transcription

I use **Whisper-medium** to convert each recording into text.

The transcript becomes an additional source of information because grammar quality is directly related to sentence structure, word usage, repetition, fillers, and grammatical errors.

The transcription stage also produces timing and decoding-quality information.

---

## 3. Text Representations

I use two pretrained language models on the generated transcripts.

### MPNet

I use `sentence-transformers/all-mpnet-base-v2` to obtain a semantic representation of the transcript.

**Representation size: 768 features**

### DeBERTa-v3-large

I use `microsoft/deberta-v3-large` to capture additional contextual and syntactic information from the transcript.

**Representation size: 1,024 features**

### Combined Deep Representation

The main high-dimensional representation combines:

| Representation | Dimensions |
|---|---:|
| WavLM-Large | 2,048 |
| WavLM-Base | 1,536 |
| MPNet | 768 |
| DeBERTa-v3-large | 1,024 |
| Whisper-medium | 2,048 |
| **Total** | **7,424** |

---

## 4. Acoustic Features

I extract **37 compact acoustic features** from every recording.

The features include:

- Duration
- RMS energy statistics
- Zero-crossing rate statistics
- Spectral centroid
- Spectral bandwidth
- Spectral rolloff
- 13 MFCC mean values
- 13 MFCC standard deviations

These features provide information about the speaker's delivery and acoustic characteristics that may complement the learned speech representations.

---

## 5. Grammar and Text Features

I build an explicit **21-feature grammar/text block**.

### Basic Transcript Features

I calculate:

- Word count
- Sentence count
- Unique word count
- Average sentence length
- Type-token ratio
- Filler count
- Repeated-word count
- Words per minute

### LanguageTool Features

I use LanguageTool to extract grammar-related statistics such as:

- Total detected issues
- Error density
- Grammar issues
- Spelling issues
- Style issues
- Punctuation-related issues
- Sentences containing errors
- Mean error length
- Replacement statistics
- Maximum error length
- Error-length variation

### CoLA Acceptability

I also use `textattack/roberta-base-CoLA` as a grammatical acceptability signal.

---

## 6. Whisper Quality Features

I extract **10 transcription-quality and timing features** from Whisper.

These include:

- Average log probability
- No-speech probability
- Compression ratio
- Speech duration
- Speech span
- Mean pause duration
- 90th-percentile pause duration
- Maximum pause duration
- Long-pause rate
- Pause ratio

These features are used as a residual correction branch rather than as the main prediction source.

---

## 7. Modeling

### Joint Ridge Model

The 7,424-dimensional representation is passed to a Ridge regression model.

I use out-of-fold predictions so that the training evaluation does not rely on direct in-sample predictions.

### Representation-Level Stack

I separately model the major representations and combine their predictions:

- WavLM-Large
- WavLM-Base
- MPNet
- DeBERTa-v3-large

The resulting representation stack is combined with the joint Ridge prediction.

### ExtraTrees Branch

I build a compact **47-feature ExtraTrees branch** containing:

- 3 model prediction features
- 8 basic transcript features
- 36 acoustic features with duration removed

This gives a nonlinear model a compact representation of the main signals.

### Grammar Residual Correction

Instead of asking the grammar features to predict the entire score independently, I train a Ridge model on the remaining error after the base ensemble.

This makes the grammar branch specialize in information not already captured by the speech and text models.

### Whisper Quality Residual Correction

I use the same residual-learning idea for the 10 Whisper quality features.

### LightGBM Auxiliary Model

I train LightGBM on the full **68-feature compact feature set**:

```text
37 acoustic
+ 21 grammar/text
+ 10 Whisper quality
= 68 features
```

The LightGBM prediction is used as a small complementary contribution in the final ensemble.

---

## 8. Final Ensemble

The final prediction combines the residual-corrected main branch with the LightGBM branch:

```text
Final Prediction
    =
0.92323656 × Champion Prediction
+
0.07676344 × LightGBM Prediction
```

Finally, predictions are clipped to the valid competition range:

```text
0 ≤ prediction ≤ 5
```

---

## 9. Validation

For the training evaluation, I use **out-of-fold predictions**.

The notebook reports:

- **RMSE**
- **Pearson correlation**

The competition requires a training RMSE to be included in the notebook, so I explicitly report the OOF training RMSE before producing the final submission.

I also include diagnostic plots for:

1. Actual score vs OOF predicted score
2. OOF residual distribution

---

## 10. Submission

The final submission is generated from the competition data itself.

I use:

- `test.csv` for the exact evaluation-row order
- `sample_submission.csv` for the required submission column names

The generated file is:

```text
submission_final.csv
```

Before saving it, I validate:

- exactly **216 rows**
- exactly **2 columns**
- correct test filename order
- no missing predictions
- no infinite predictions
- predictions remain within `[0, 5]`

---

## Repository Structure

```text
shl-hiring-assessment-2026/
│
├── grammar_score.ipynb
├── submission_final.csv
├── README.md
├── requirements.txt
├── .gitignore
└── Data/                  # local / competition data, not committed
```

The competition dataset is not included in the repository. It should be attached through the Kaggle competition environment.

---

## Notebook

The complete implementation is available in:

**[`grammar_score.ipynb`](grammar_score.ipynb)**

The notebook contains:

- dataset loading
- audio preprocessing
- Whisper transcription
- acoustic feature extraction
- grammar feature extraction
- deep representation extraction
- OOF modeling
- ensemble construction
- training evaluation
- diagnostic visualizations
- final submission generation and validation

---

## Submission File

The latest prediction file is:

**[`submission_final.csv`](submission_final.csv)**

It contains the predictions for the 216 evaluation samples.

---

## Reproducibility

To reproduce the pipeline:

1. Open the notebook in Kaggle.
2. Attach the **SHL Hiring Assessment 2026** competition dataset.
3. Enable a GPU runtime for the pretrained speech and language models.
4. Run the notebook from the beginning.
5. The intermediate representations are cached under `/kaggle/working`.
6. The final file is created at:

```text
/kaggle/working/submission_final.csv
```

---

## Important Notes

The project uses pretrained models only as feature extractors. The final prediction system is built from the extracted representations and explicit feature blocks described in this README.

The reported **0.3437** is my latest Kaggle submission score. It is the external leaderboard result for this challenge.

The repository does not contain the competition audio dataset or large generated model artifacts.

---

## Technologies

```text
Python
PyTorch
Transformers
faster-whisper
Librosa
LanguageTool
scikit-learn
LightGBM
NumPy
Pandas
Matplotlib
```

---

## Final Result

My latest Kaggle submission achieved:

> **0.3437 Kaggle Score**

This project combines speech representations, transcript representations, explicit grammar analysis, acoustic information, and ensemble modeling to build a practical spoken grammar scoring system.
