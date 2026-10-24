Notebook protocol (dropna, description only, random 70/30, val = last 10 % of train), mean ± std over seeds

| model   |   n |   acc_best |   acc_best_std |   acc_last |   acc_last_std |   best_epoch |
|:--------|----:|-----------:|---------------:|-----------:|---------------:|-------------:|
| cnn     |   5 |     0.9782 |         0.0029 |     0.9777 |         0.0024 |       9.2000 |
| cnn_bug |   5 |     0.9460 |         0.0049 |     0.9369 |         0.0058 |       5.0000 |
| gru     |   5 |     0.9630 |         0.0104 |     0.9592 |         0.0067 |       3.0000 |
| lstm    |   5 |     0.9761 |         0.0040 |     0.9764 |         0.0034 |       3.0000 |