# Results

Test-set outputs of the final MSAF-DenseNet201 checkpoint (best validation epoch 59, raw weights, 4-way flip TTA), evaluated on the 918-image held-out split.

| File | Contents |
|---|---|
| `summary_metrics.json` / `.csv` | Accuracy, balanced accuracy, weighted and macro precision/recall/F1, Cohen's κ, MCC, ROC-AUC, log-loss |
| `per_class_metrics.csv` | Precision, recall, F1 and support per class |
| `confusion_matrix.csv` / `.png` | Test-set confusion matrix (rows are true labels) |
| `confusion_matrix_normalised.png` | Row-normalised confusion matrix |
| `training_curves.png` | Loss, train/val/EMA accuracy and learning rate over all 100 epochs |
| `history.json` | Per-epoch record: loss, train_acc, val_acc, val_acc_ema, lr |
| `test_probs.npy` / `test_true.npy` | Class probabilities and true labels for every test image, for re-computing any metric |
| `Final_30_Model_Ranked_Results.csv` | All 30 models ranked by test accuracy (ties broken by κ) |

Headline: **accuracy 0.9150 · F1 (weighted) 0.9153 · F1 (macro) 0.9304 · κ 0.8877 · MCC 0.8885 · ROC-AUC 0.9892**
