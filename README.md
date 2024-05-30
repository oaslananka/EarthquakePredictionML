# Earthquake Prediction using Machine Learning

This project aims to predict earthquake magnitudes and occurrences using machine learning models.

## Project Structure

```ini {"id":"01HZ4MWPV1WXE217FHBXMVKS0E"}
earthquake-prediction-ml/
├── data/
│   └── earthquake_data.csv
├── src/
│   └── main.py
├── README.md
├── LICENSE
└── requirements.txt
```

## Installation

1. Clone the repository:

```sh {"id":"01HZ4MWPV22SCECAA8RZ3TFA9H"}
git clone https://github.com/oaslananka/earthquake-prediction-ml.git
cd earthquake-prediction-ml
```

2. Create and activate a virtual environment:

```sh {"id":"01HZ4MWPV22SCECAA8S2APFYT9"}
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scriptsctivate`
```

3. Install the dependencies:

```sh {"id":"01HZ4MWPV22SCECAA8S2CKJ758"}
pip install -r requirements.txt
```

## Usage

1. Run the main script:

```sh {"id":"01HZ4MWPV22SCECAA8S41S7ZCC"}
python src/main.py
```

2. Explore the data using Jupyter notebooks:

```sh {"id":"01HZ4MWPV22SCECAA8S76FHAPA"}
jupyter notebook notebooks/exploration.ipynb
```

## Data Source

The earthquake data is fetched from the USGS (United States Geological Survey) API. The data includes information about earthquake magnitudes, locations, depths, and times.

## Algorithms and Methods

This project uses several machine learning algorithms for earthquake prediction, including:

- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

The models are trained using features engineered from the raw earthquake data, such as statistical measures, zero-crossings, peak counts, FFT values, and the Hilbert transform's amplitude envelope.

## Example Project

This project serves as an example of using machine learning techniques for predicting natural events. It demonstrates data fetching, preprocessing, feature engineering, model training, and prediction.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
