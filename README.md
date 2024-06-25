# Blockchain Address Classifier

This project provides tools to scrape blockchain addresses, process datasets, train a machine learning model to classify blockchain addresses, and perform predictions on new addresses.

## Installation

1. Clone this repository:

   ```
   git clone https://github.com/rayaanoidprime/blockchain-address-classifier.git
   cd blockchain-address-classifier

   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Data Collection and Preprocessing

If you need to collect new data or process raw datasets, follow these steps:

1. Scrape blockchain addresses (optional):

   ```
   python scraper.py sol --frequency 5 --delay 120 # for solana
   python scraper.py oklink # for btc, eth, avax, bsc etc
   ```

   Modify the parameters as needed.

2. Process the scraped datasets (optional):

```
   python process_scrapped_datasets.py --eth_input /path/to/eth_addresses.csv --btc_input /path/to/bitcoin_address.csv --output_dir /path/to/output/directory
```

This script processes Ethereum ,Bitcoin , and all other csv address datasets that have been scrapped and makes it into a combines csv. Adjust the input paths as necessary.

## Model Training and Prediction

### Training the Model (optional)

To train the blockchain address classification model:

```
python model.py train --input path_to_your_input.csv --model model_output_path.joblib
```

This will train the model, save it to the specified path, and generate an OOB error plot.

- `--input`: Path to your preprocessed CSV file containing addresses and their corresponding blockchains.
- `--model`: Path where you want to save the trained model.

For Batch prediction of a lot of addresses :

## Making Predictions

### To predict the blockchain for a given address:

```
python model.py predict --input 0x742d35Cc6634C0532925a3b844Bc454e4438f44e --model path_to_your_model.joblib
```

### To predict a batch of a lot of addresses in a txt file. The file format should be one address per line:

```
python model.py predict_batch --input path_to_addresses.txt --model path_to_your_model.joblib --output predictions_output.csv

```

- `--input`: The blockchain address you want to classify.
- `--model`: Path to your trained model file.
- `--output`: Path where you want to save the CSV file with predictions.

This will process all addresses in the input file and create a CSV file with two columns: Address and Predicted_Blockchain.

## File Descriptions

- `scraper.py`: Script for scraping blockchain addresses.
- `process_scrapped_datasets.py`: Script for processing raw datasets.
- `model.py`: Script for training the classification model and making predictions.
- `requirements.txt`: List of Python dependencies.
