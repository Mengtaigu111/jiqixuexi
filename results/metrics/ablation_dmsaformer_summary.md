# DMSAFormer module ablation (raw test metrics, mean/std over seeds)

| variant | horizon | trainable_params | MSE mean | MSE std | MAE mean | MAE std | Runs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| no_window_norm | 90 | 393794 | 169213.59 | 2132.18 | 318.65 | 2.65 | 5 |
| no_variable_attention | 90 | 393794 | 186721.01 | 4674.71 | 337.27 | 5.37 | 5 |
| no_decomposition | 90 | 393794 | 187527.56 | 7516.44 | 338.83 | 9.61 | 5 |
| full | 90 | 393794 | 192194.49 | 4952.14 | 343.30 | 5.48 | 5 |
| no_correction | 90 | 244340 | 195328.29 | 5623.40 | 347.60 | 6.93 | 5 |
| full | 365 | 532669 | 428501.54 | 31780.65 | 524.15 | 21.29 | 5 |
| no_variable_attention | 365 | 532669 | 431982.91 | 35649.80 | 525.94 | 23.12 | 5 |
| no_window_norm | 365 | 532669 | 439905.36 | 14696.40 | 536.93 | 10.91 | 5 |
| no_correction | 365 | 347740 | 447761.56 | 8338.52 | 538.82 | 6.30 | 5 |
| no_decomposition | 365 | 532669 | 501846.97 | 20543.93 | 571.75 | 12.77 | 5 |
