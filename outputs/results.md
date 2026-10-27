| Images | Accuracy | Precision (fake) | Recall (fake) |
|---|---|---|---|
| clean | 0.875 | 0.849 | 0.914 |
| JPEG q=30 | 0.868 | 0.847 | 0.899 |

Confusion matrix, clean (rows = true, columns = predicted):

| | pred fake | pred real |
|---|---|---|
| true fake | 920 | 87 |
| true real | 163 | 830 |

Confusion matrix, JPEG q=30 (rows = true, columns = predicted):

| | pred fake | pred real |
|---|---|---|
| true fake | 905 | 102 |
| true real | 163 | 830 |
