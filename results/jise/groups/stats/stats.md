Query-masked groups, OOF predictions on all 11,612 groups.

## Paired comparisons

diff = acc(A) − acc(B); CI = cluster bootstrap over duplicate groups (B=2000); McNemar exact, Holm-adjusted over all pairs; folds_A_better = of 5 folds.

| A                        | B                             |   acc_A |   acc_B |      diff |   diff_ci_lo |   diff_ci_hi |   A_right_B_wrong |   A_wrong_B_right |   mcnemar_p_holm |   folds_A_better |
|:-------------------------|:------------------------------|--------:|--------:|----------:|-------------:|-------------:|------------------:|------------------:|-----------------:|-----------------:|
| e5-base fine-tuned, ens5 | e5-base frozen + LR           |  0.9775 |  0.9684 |  0.009128 |     0.00656  |     0.01163  |               157 |                51 |        4.982e-13 |                5 |
| e5-base fine-tuned, ens5 | TF-IDF + LR                   |  0.9775 |  0.9626 |  0.0149   |     0.01176  |     0.0181   |               265 |                92 |        1.192e-19 |                5 |
| e5-base fine-tuned, ens5 | CNN, ens5                     |  0.9775 |  0.9656 |  0.01188  |     0.008695 |     0.01516  |               242 |               104 |        4.982e-13 |                5 |
| e5-base fine-tuned, ens5 | qwen2.5-7b, 4-shot            |  0.9775 |  0.9384 |  0.0391   |     0.03489  |     0.04333  |               536 |                82 |        1.193e-81 |                5 |
| e5-base fine-tuned, ens5 | qwen3-8b, 4-shot              |  0.9775 |  0.9374 |  0.04013  |     0.03598  |     0.04457  |               561 |                95 |        2.2e-80   |                5 |
| e5-base fine-tuned, ens5 | rubert-tiny2 fine-tuned, ens5 |  0.9775 |  0.9674 |  0.01016  |     0.007662 |     0.01264  |               173 |                55 |        1.484e-14 |                5 |
| e5-base frozen + LR      | TF-IDF + LR                   |  0.9684 |  0.9626 |  0.00577  |     0.00267  |     0.008816 |               199 |               132 |        0.001095  |                5 |
| e5-base frozen + LR      | CNN, ens5                     |  0.9684 |  0.9656 |  0.002756 |    -0.000603 |     0.006213 |               215 |               183 |        0.2402    |                4 |
| TF-IDF + LR              | CNN, ens5                     |  0.9626 |  0.9656 | -0.003014 |    -0.005939 |     0.000172 |               144 |               179 |        0.175     |                0 |
| qwen2.5-7b, 4-shot       | qwen3-8b, 4-shot              |  0.9384 |  0.9374 |  0.001033 |    -0.002498 |     0.00458  |               234 |               222 |        0.6065    |                3 |

## LLM temperature scaling (T fitted on inner validation of the training folds)

| model                     |   T_mean |   T_std |   ece_before_mean |   ece_before_std |   ece_after_mean |   ece_after_std |   nll_before_mean |   nll_before_std |   nll_after_mean |   nll_after_std |
|:--------------------------|---------:|--------:|------------------:|-----------------:|-----------------:|----------------:|------------------:|-----------------:|-----------------:|----------------:|
| qwen2.5-7b-instruct_0shot |  10.0284 |  0.6286 |            0.2340 |           0.0059 |           0.0799 |          0.0105 |            2.2589 |           0.0665 |           0.5043 |          0.0050 |
| qwen2.5-7b-instruct_4shot |   2.9193 |  0.2375 |            0.0481 |           0.0131 |           0.0431 |          0.0481 |            0.4092 |           0.0990 |           0.2256 |          0.0345 |
| qwen3-8b_0shot            |   9.9387 |  0.4540 |            0.1882 |           0.0040 |           0.0694 |          0.0075 |            2.2126 |           0.0444 |           0.4381 |          0.0044 |
| qwen3-8b_4shot            |   4.9479 |  0.3000 |            0.0588 |           0.0126 |           0.0218 |          0.0109 |            0.6159 |           0.1402 |           0.1939 |          0.0282 |

Note: for 4-shot the validation predictions were produced with the few-shot examples of their own fold; for 0-shot the prompt is identical across folds.
