# Tanzanian Water Pump Maintenance

A data science project exploring how machine learning can help prioritize water pump inspections and maintenance in Tanzania.

## Objective

Predict which water pumps are functional or may require intervention, then translate these predictions into practical inspection priorities. The goal is to support maintenance planning, not to automate repair or replacement decisions.

## Data

This project uses the [Pump it Up: Data Mining the Water Table](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/) dataset hosted by DrivenData.

Download the competition data and place the original CSV files in `data/raw/`. The original dataset is not included in this repository.

**Attribution**

DrivenData. (2015). *Pump it Up: Data Mining the Water Table.* https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/

See the [competition paper](https://arxiv.org/abs/1606.07781) for additional background.

## Methodology

### Exploratory data analysis

The exploratory analysis examines pump functionality across regions and infrastructure characteristics, as well as missing values, recorded population, and estimated pump age.

Some notable findings include:

- Recorded pump age has a median of approximately 13 years among records with a valid age estimate.
- Population is recorded as zero for roughly 36% of pumps, making it an uncertain proxy for the number of people affected.
- Functionality rates vary substantially between regions and waterpoint types.

These are descriptive findings from historical records, not evidence of causal relationships or current pump conditions.

### Data preparation

- Split the labelled data into training, validation, and test subsets using stratified sampling (60% / 20% / 20%, random seed 42).
- Impute missing numerical values and add missingness indicators.
- Impute missing categorical values and one-hot encode categorical features for the scikit-learn models.
- Derive recorded year, recorded month, and estimated pump age from the available date and construction-year fields.
- Treat invalid or unavailable construction years as missing age information.

### Model development

The project compares a majority-class baseline with Random Forest and CatBoost models.

Two prediction tasks are considered:

1. **Three-class classification:** predict whether a pump is functional, functional but needs repair, or non-functional.
2. **Binary screening:** identify pumps that are either non-functional or in need of repair, versus those recorded as functional.

Balanced accuracy, precision, recall, and F1-score are used alongside ordinary accuracy to assess performance. Permutation importance is also used to explore which features contribute most to model predictions.

## Results

The following results are from the held-out evaluation subset.

### Binary screening

The binary task groups pumps recorded as *functional needs repair* or *non functional* into a single intervention-needed category.

| Model | Accuracy | Balanced accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|---:|
| Majority baseline | 54.3% | 50.0% | 0.0% | 0.0% | 0.0% |
| Random Forest | 81.6% | 81.2% | 82.1% | 76.3% | 79.1% |
| CatBoost | 82.2% | 81.9% | 81.9% | 78.3% | 80.1% |

CatBoost performs slightly better overall on the binary task, while Random Forest provides a competitive and relatively straightforward alternative. The difference is modest, so model choice should also consider runtime, maintainability, and operational requirements.

The three-class models achieve approximately 76% accuracy with Random Forest and 75% with CatBoost. Their predictions can also be collapsed into the binary screening task, although the separate repair-needed class is more difficult to distinguish reliably.

These metrics measure agreement with historical recorded labels; they do not establish how well the models identify pumps that currently require intervention.

## Intervention prioritization

The prioritization module applies a trained binary classifier to unlabelled pump records and produces ranked CSV files for further investigation.

It explores three approaches:

- **Risk-based:** rank pumps by their predicted probability of needing intervention.
- **Risk and geographical scarcity:** combine predicted intervention risk with an estimate of how few nearby functional alternatives are available, using a 2 km neighbourhood radius.
- **Risk and population:** combine predicted intervention risk with the logarithm of positive recorded population as a rough proxy for potential impact.

The resulting rankings are intended to help planners identify pumps for inspection. They are not definitive repair schedules.

## Limitations and considerations

- **Historical data:** the records date primarily from 2011–2013. Predictions should not be interpreted as the current condition of a pump.
- **Recorded labels:** the models predict recorded functionality categories, not the underlying cause of failure or the repair required.
- **Population data:** zero, missing, or inaccurate population values can distort impact rankings. Population-based rankings should be reviewed rather than interpreted as exact estimates of people affected.
- **Geographical data:** invalid or placeholder coordinates, including records at `(0, 0)`, can distort neighbourhood counts and geographical scarcity scores. These rankings require additional geographic validation.
- **Correlated features:** related variables, such as `quantity` and `quantity_group`, may share predictive information and affect feature-importance interpretation.
- **Operational constraints:** repair costs, travel time, spare-parts availability, and maintenance capacity are not incorporated. These would be needed for a more complete operational prioritization strategy.

## Reproducibility

The project uses Python and the dependencies listed in `requirements.txt`.

### 1. Create and activate a virtual environment

From the project root, run:

```bash
python -m venv .venv
source .venv/bin/activate
```

This activation command is for Linux and macOS.

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Add the data

Download the required competition CSV files and place them in `data/raw/`, following the filenames expected by the data-loading module:
- `training_set_labels`
- `training_set_values`

### 4. Run the project

From the project root, run:

```bash
python -m src.water_pumps.main
```

Generated analysis outputs, evaluation results, and prioritization files are saved to the project's configured output directories.

## Project status

This is an initial prototype demonstrating exploratory analysis, predictive modelling, model evaluation, and risk-based intervention prioritization. Further work should validate the geographical and population-based rankings, assess model performance across relevant subgroups, and incorporate operational constraints before practical deployment.