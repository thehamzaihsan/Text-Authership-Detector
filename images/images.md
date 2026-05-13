# Image Directory

This directory contains all chart images generated from `test.ipynb` and `ded_executed.ipynb`.

---

## test.ipynb (4-Author Models)

### Dataset Overview
| File | Description |
|------|-------------|
| `test_dataset_summary_table.png` | Summary table of the 4 filtered senders — message count, avg word count, avg char count per author |

### Model 1: Word BOW + Naive Bayes (78.83% acc)
| File | Description |
|------|-------------|
| `test_bow_nb_per_sender_metrics.png` | Per-sender grouped bar chart of Precision, Recall, F1-Score |
| `test_bow_nb_confusion_matrix.png` | Normalized confusion matrix heatmap |
| `test_bow_nb_test_support.png` | Horizontal bar chart of test set support per sender |

### Model 2: Word TF-IDF + Naive Bayes (79.75% acc)
| File | Description |
|------|-------------|
| `test_tfidf_nb_per_sender_metrics.png` | Per-sender grouped bar chart of Precision, Recall, F1-Score |
| `test_tfidf_nb_confusion_matrix.png` | Normalized confusion matrix heatmap |
| `test_tfidf_nb_test_support.png` | Horizontal bar chart of test set support per sender |

### Model 3: Char N-Grams + TF-IDF + Naive Bayes (75.08% acc)
| File | Description |
|------|-------------|
| `test_char_tfidf_nb_per_sender_metrics.png` | Per-sender grouped bar chart of Precision, Recall, F1-Score |
| `test_char_tfidf_nb_confusion_matrix.png` | Normalized confusion matrix heatmap |
| `test_char_tfidf_nb_test_support.png` | Horizontal bar chart of test set support per sender |

### Model 4: Char N-Grams + TF-IDF + Logistic Regression (77.75% acc)
| File | Description |
|------|-------------|
| `test_char_tfidf_logreg_per_sender_metrics.png` | Per-sender grouped bar chart of Precision, Recall, F1-Score |
| `test_char_tfidf_logreg_confusion_matrix.png` | Normalized confusion matrix heatmap |
| `test_char_tfidf_logreg_test_support.png` | Horizontal bar chart of test set support per sender |

### Model 5: Word BOW + Handcrafted Features + Naive Bayes (78.67% acc)
| File | Description |
|------|-------------|
| `test_bow_handcrafted_nb_per_sender_metrics.png` | Per-sender grouped bar chart of Precision, Recall, F1-Score |
| `test_bow_handcrafted_nb_confusion_matrix.png` | Normalized confusion matrix heatmap |
| `test_bow_handcrafted_nb_test_support.png` | Horizontal bar chart of test set support per sender |

### Model 6: Ensemble + Model Comparison (79.67% acc)
| File | Description |
|------|-------------|
| `test_model_comparison.png` | Horizontal bar chart comparing all 6 model accuracies |
| `test_ensemble_confusion_matrix.png` | Normalized confusion matrix heatmap for the ensemble model |

---

## ded_executed.ipynb (Deductive 9-Class Model)

### Dataset Overview
| File | Description |
|------|-------------|
| `ded_dataset_summary_table.png` | Summary table of the top 8 senders used as known authors |

### Deductive Model: Char TF-IDF + Logistic Regression — 9-Class (67.23% acc)
| File | Description |
|------|-------------|
| `ded_9class_per_sender_metrics.png` | Per-class grouped bar chart of Precision, Recall, F1-Score for all 9 classes (8 known + UNKNOWN) |
| `ded_9class_confusion_matrix.png` | Normalized confusion matrix heatmap for the 9-class model |

---

## Summary

| Source | Images |
|--------|--------|
| `test.ipynb` | 18 charts (6 models × per-sender metric + confusion matrix + support, plus model comparison and ensemble confusion matrix) |
| `ded_executed.ipynb` | 3 charts (dataset table, per-class metrics, confusion matrix) |
| **Total** | **21 images** |
