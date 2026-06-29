Goal: Implement and run the third improved model for the household power consumption forecasting course project.

We only need to implement the custom improved model, not the baseline LSTM or baseline Transformer.

Task background:
The project is a multivariate time series forecasting task for household electricity consumption. The model should use the past 90 days of data to predict future daily `global_active_power`. There are two separate forecasting settings:

1. Short-term forecasting:

   * input length: 90 days
   * prediction length: 90 days

2. Long-term forecasting:

   * input length: 90 days
   * prediction length: 365 days

The short-term and long-term models must be trained separately. Do not reuse the trained parameters from one task for the other.

Target variable:

* `global_active_power`

Input variables:
Use all available processed daily features, including but not limited to:

* `global_active_power`
* `global_reactive_power`
* `voltage`
* `global_intensity`
* `sub_metering_1`
* `sub_metering_2`
* `sub_metering_3`
* `sub_metering_remainder`, if it can be computed
* weather variables such as `RR`, `NBJRR1`, `NBJRR5`, `NBJRR10`, `NBJBROU`
* optional calendar features such as day of week, month, weekend flag

Data processing requirements:

1. Load `train.csv` and test data. The test file may be named `test.csv` or `tes.csv`; support both if possible.
2. Parse date/time columns automatically.
3. If the raw data is minute-level, aggregate it to daily level:

   * `global_active_power`: daily sum
   * `global_reactive_power`: daily sum
   * `sub_metering_1`: daily sum
   * `sub_metering_2`: daily sum
   * `sub_metering_3`: daily sum, if available
   * `voltage`: daily mean
   * `global_intensity`: daily mean
   * weather variables such as `RR`, `NBJRR1`, `NBJRR5`, `NBJRR10`, `NBJBROU`: take the daily value or first valid value of the day
4. Handle missing values safely:

   * sort by date
   * interpolate numeric missing values when reasonable
   * otherwise use forward fill and backward fill
5. Compute:
   `sub_metering_remainder = global_active_power * 1000 / 60 - (sub_metering_1 + sub_metering_2 + sub_metering_3)`
   if all required columns exist.
6. Scale features using only the training data. Do not fit scalers on test data.
7. Create sliding-window samples:

   * X shape: `[num_samples, 90, num_features]`
   * y shape for short-term: `[num_samples, 90]`
   * y shape for long-term: `[num_samples, 365]`

Model to implement:
Implement a custom model named:

DMSAFormer: Decomposition-based Multi-Scale Attention Transformer

The model should contain the following components:

1. Series decomposition module:

   * Decompose the input sequence into trend and residual components.
   * Use a moving average layer over the time dimension.
   * `trend = MovingAverage(x)`
   * `residual = x - trend`

2. Linear trend branch:

   * Use the trend component of the target variable or trend features to predict the future trend.
   * This branch should be simple, for example Linear or small MLP.
   * Output shape: `[batch_size, pred_len]`

3. Multi-scale residual branch:

   * Use the residual component as input.
   * Apply multiple 1D convolution layers with different kernel sizes.
   * Recommended kernels:

     * kernel size 3 for short-term fluctuations
     * kernel size 7 for weekly pattern
     * kernel size 30 for monthly pattern
   * Concatenate the outputs of the multi-scale convolution layers.

4. Optional but preferred variable attention module:

   * Learn feature-wise importance weights for the multivariate input.
   * Use a small MLP or linear layer followed by softmax or sigmoid.
   * Multiply input features by the learned weights.
   * This is meant to make the model aware of which variables are more useful for forecasting.

5. Transformer encoder:

   * Feed the multi-scale residual representation into a Transformer encoder.
   * Use batch-first format if possible.
   * Recommended default settings:

     * hidden dimension: 64 or 128
     * number of attention heads: 4
     * number of encoder layers: 2
     * dropout: 0.1

6. Prediction head:

   * The Transformer residual branch outputs a residual prediction.
   * The final prediction should be:
     `y_pred = y_trend + y_residual`
   * Output shape must be `[batch_size, pred_len]`.

Training requirements:

1. Train two separate DMSAFormer models:

   * one with `pred_len=90`
   * one with `pred_len=365`
2. Run at least 5 independent experiments for each prediction length using different random seeds, for example:

   * 2026, 2027, 2028, 2029, 2030
3. For each run, record:

   * test MSE
   * test MAE
4. After 5 runs, report:

   * MSE mean
   * MSE std
   * MAE mean
   * MAE std
5. Save results to:

   * `results/dmsaformer_90_results.csv`
   * `results/dmsaformer_365_results.csv`
   * `results/summary.csv`

Training details:

* Use PyTorch.
* Use Adam or AdamW optimizer.
* Recommended learning rate: 1e-3.
* Recommended batch size: 32.
* Recommended epochs: 50 to 100.
* Use early stopping based on validation loss if validation split is available.
* Use MSE loss for training.
* Evaluation should compute both MSE and MAE on the original or consistently scaled target. If predictions are inverse-transformed, evaluate in original scale.

Plotting requirements:
For each prediction setting, save at least one prediction comparison plot:

1. `figures/dmsaformer_90_prediction.png`
2. `figures/dmsaformer_365_prediction.png`

Each plot should show:

* Ground Truth curve
* Predicted curve
* clear title
* x-axis as future day index
* y-axis as global_active_power

Code structure:
Please organize the code clearly. Suggested structure:

* `data_utils.py`

  * load data
  * aggregate to daily level
  * handle missing values
  * scale features
  * create sliding-window datasets

* `models/dmsaformer.py`

  * MovingAverage
  * SeriesDecomposition
  * VariableAttention
  * DMSAFormer

* `train_dmsaformer.py`

  * command-line training script
  * supports `--pred_len 90` and `--pred_len 365`
  * supports `--seed`
  * saves metrics and plots

* `run_experiments.py`

  * runs 5 seeds for pred_len=90
  * runs 5 seeds for pred_len=365
  * summarizes mean and std

Command-line examples:

```bash
python train_dmsaformer.py --pred_len 90 --seed 2026
python train_dmsaformer.py --pred_len 365 --seed 2026
python run_experiments.py
```

Implementation notes:

1. Make the code robust to different column names for date/time.
2. Print the final feature list used for training.
3. Print the shapes of train, validation, and test datasets.
4. Make sure the output length exactly matches `pred_len`.
5. Make sure the short-term model and long-term model are trained separately.
6. Add comments explaining the model design:

   * decomposition handles trend and residual separately
   * multi-scale convolution captures 3-day, 7-day, and 30-day patterns
   * variable attention learns the importance of electricity and weather variables
   * Transformer encoder models long-range dependencies
   * linear trend branch improves long-term stability

Expected final output:
After running the code, I should get:

1. trained DMSAFormer results for 90-day forecasting
2. trained DMSAFormer results for 365-day forecasting
3. MSE and MAE for each of 5 runs
4. mean and std of MSE and MAE
5. prediction-vs-ground-truth plots
6. clean code that can be linked in the project GitHub repository
