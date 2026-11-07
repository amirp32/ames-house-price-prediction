# Ames Housing Price Prediction

**Exploratory data analysis, missing-value treatment, and a Linear Regression baseline for house sale prices.**

This Python portfolio project explores the Ames Housing dataset and builds a regression model to estimate `SalePrice` from property characteristics. The notebook covers data inspection, missingness visualization, feature selection, imputation, categorical encoding, and model evaluation.

## Project overview

| Item | Details |
| --- | --- |
| Notebook | [Housing.ipynb](Housing.ipynb) |
| Input file | `Ames_Housing_Data.csv` |
| Initial dataset | 2,930 rows and 81 columns, according to saved outputs |
| Target | `SalePrice` |
| Model | Scikit-learn `LinearRegression` |
| Evaluation split | 80% training / 20% testing, `random_state=42` |

Only Linear Regression is implemented in the supplied notebook. Random Forest and Gradient Boosting are possible extensions, not completed experiments.

## Dataset and features

The notebook expects a CSV named **`Ames_Housing_Data.csv`** in the same working directory. Example predictors include:

- Property quality and condition: `Overall Qual`, `Overall Cond`, exterior and kitchen quality.
- Size and layout: `Gr Liv Area`, floor areas, room counts, and bathrooms.
- Construction history: `Year Built`, `Year Remod/Add`, and `Garage Yr Blt`.
- Location and property type: `Neighborhood`, zoning, building type, and house style.
- Amenities and structures: basement and garage characteristics, fireplaces, decks, and porches.

The CSV was not included in the supplied project files. An authorized copy with the expected columns is required to rerun the notebook. Saved outputs and figures can be inspected without it. The supplied materials do not identify a download URL or dataset license.

## Analysis workflow

1. **Inspect the data:** Review records, data types, numerical summaries, and missing-value counts.
2. **Explore missingness:** Plot missing values and inspect selected records with incomplete basement, garage, and electrical information.
3. **Select features:** Drop five explicitly named columns (`Pool QC`, `Misc Feature`, `Alley`, `Fence`, `Fireplace Qu`) and eight additional columns selected for exclusion during exploration.
4. **Handle incomplete records:** Drop rows missing selected fields; apply mean imputation to selected numerical fields and most-frequent imputation to selected categorical fields.
5. **Encode categories:** Use `pd.get_dummies()` to transform object-type columns into numeric features.
6. **Train and evaluate:** Fit Linear Regression, calculate MSE, MAE, and R², then visualize actual versus predicted prices and residuals.

The notebook also contains categorical count plots, a correlation heatmap, and a numerical box-plot loop. The saved output shows that the early numeric-column selection is empty, so that loop needs correction before claiming a completed numerical box-plot analysis.

## Recorded results

The following metrics come from the **saved notebook outputs**. They have not been independently reproduced for this README because the source CSV was not supplied.

| Metric | Recorded value |
| --- | ---: |
| R² | 0.8631 |
| Mean Absolute Error (MAE) | 15,763.93 |
| Mean Squared Error (MSE) | 963,287,709.15 |

MAE is expressed in the same units as `SalePrice`; MSE uses squared units. R² is a regression goodness-of-fit measure and should not be described as “86.31% prediction accuracy.”

These metrics describe the existing notebook run. **The preprocessing assignment issue described below must be fixed and the model reevaluated before using them as the results of a corrected workflow.**

## Run the notebook

Install the required packages and launch Jupyter:

```bash
python -m pip install jupyter pandas numpy matplotlib seaborn scikit-learn
jupyter notebook
```

1. Download `Housing.ipynb` and place it in a project folder.
2. Place `Ames_Housing_Data.csv` in the same folder.
3. Open the notebook in Jupyter and review the implementation notes below.
4. Run the cells in order, starting with the imports and CSV-loading cell.

For Google Colab, upload both the notebook and CSV, and ensure the CSV is in the runtime's working directory before running `pd.read_csv()`.

No pinned dependency file was supplied. The installation command lists required packages but does not define a verified reproduction environment.

## Implementation notes and limitations

### Numerical imputation assignment

The numerical-imputation cell selects:

```python
['Lot Frontage', 'Mas Vnr Area', 'BsmtFin SF 2',
 'Total Bsmt SF', 'Garage Yr Blt']
```

but assigns the transformed values to a list containing **`Bsmt Unf SF` instead of `Total Bsmt SF`** in the fourth position. This overwrites unfinished basement area with total basement area, including values in rows that were not missing.

Use one consistent column list for selection and assignment, then rerun preprocessing and evaluation. The supplied notebook has not been modified as part of this README update.

### Evaluation design

- **Preprocessing before splitting:** Imputation, categorical encoding, and feature-selection exploration occur before the train/test split. Fit learned transformations on training data only, preferably within a pipeline and cross-validation folds.
- **Identifier retained:** `PID` remains a predictor. Review and normally exclude this parcel identifier when building a model intended to generalize to new properties.
- **Missing-value meaning:** Missing basement or garage values may indicate an absent structure rather than an unknown measurement. Confirm this with a data dictionary before replacing values with means or modes.
- **Feature-selection comments:** The notebook labels its first five dropped columns as having more than 50% missing values, but performs an explicit drop rather than calculating a threshold. Treat this as a manual selection requiring verification.
- **One evaluation split:** No cross-validation, baseline comparison, hyperparameter search, or time-based validation is implemented.
- **Prediction scope:** There is no deployed application, saved model artifact, or reusable inference pipeline in the supplied notebook. Results on this dataset do not establish performance on other housing markets.

### Plotting and compatibility

- The early `int_columns` selection produces an empty list in the saved output. Select numerical columns explicitly with `df.select_dtypes(include='number').columns` when repairing this section.
- The correlation cell uses `df.corr()` on a mixed-type dataframe. Explicitly selecting numerical columns avoids relying on version-dependent defaults.
- The residual plot uses the legacy `sns.distplot` call, which may need updating for the installed Seaborn version.
- Imputation cells show `SettingWithCopyWarning` in the saved outputs. Use an explicit dataframe copy after filtering and clear column assignments when revising the notebook.

## Next improvements

- Fix the numerical-imputation mapping and numeric-column selection, then rerun the notebook from a clean kernel.
- Document dataset provenance, variable meanings, and missing-value conventions.
- Build a `ColumnTransformer` and `Pipeline` for preprocessing and model training.
- Compare against a median-price baseline and evaluate with cross-validation.
- Explore Ridge, Lasso, Random Forest, and Gradient Boosting as additional experiments.
- Inspect residuals by price range and neighborhood, and test a log-transformed target.
- Add a tested dependency file and save the complete fitted preprocessing-and-model pipeline.

## Skills demonstrated

Python · Pandas · NumPy · Exploratory Data Analysis · Missing-Value Analysis · Data Visualization · Feature Selection · Imputation · One-Hot Encoding · Scikit-learn · Linear Regression · Regression Evaluation

## Project scope

This repository presents an educational housing-price modeling notebook. Its purpose is to demonstrate data preparation and regression analysis, with implementation issues and evaluation limits documented for further improvement.
