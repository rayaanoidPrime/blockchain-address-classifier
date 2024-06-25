import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import re
from collections import Counter
import matplotlib.pyplot as plt
import hashlib 
import cb58ref
import joblib
import argparse

class CryptoAddressFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for address in X:
            features.append(self.engineer_features(address))
        column_names = [
            'length', 'uppercase_ratio', 'lowercase_ratio', 'digit_ratio',
            'starts_with_0x', 'starts_with_1', 'starts_with_3', 'starts_with_bc1', 'starts_with_T',
            'entropy', 'is_eth_address', 'is_btc_address', 'is_trx_address', 'is_sol_address',
            'most_common_bigram_count', 'mean_ascii', 'median_ascii', 'ascii_range','first_digit_position', 'eth_bsc_checksum_valid', 'avax_checksum_valid'
        ]
        return pd.DataFrame(features, columns=column_names)

    def eth_bsc_checksum_valid(self,address):
      from eth_utils import to_checksum_address, is_checksum_address
      try:
          return int(is_checksum_address(to_checksum_address(address)))
      except:
          return 0

    def avax_checksum_valid(self,address):
      try:
          # Remove the chain prefix if present
          if address.startswith('X-'):
              address = address[2:]
          
          # Decode the CB58 address
          decoded = cb58ref.cb58decode(address)
          
          # The last 4 bytes are the checksum
          without_checksum = decoded[:-4] #payload
          checksum = decoded[-4:]
          calculated_checksum = hashlib.sha256(without_checksum).digest()[-4:]
          
          if checksum == calculated_checksum:
              return 1
          else:
              return 0
      except:
          return 0

    def engineer_features(self, address):
        features = {}

        # Address length
        features['length'] = len(address)

        # Character frequency
        char_counts = Counter(address)
        features['uppercase_ratio'] = sum(char_counts[c] for c in char_counts if c.isupper()) / len(address)
        features['lowercase_ratio'] = sum(char_counts[c] for c in char_counts if c.islower()) / len(address)
        features['digit_ratio'] = sum(char_counts[c] for c in char_counts if c.isdigit()) / len(address)

        # Prefix analysis
        features['starts_with_0x'] = int(address.startswith('0x'))
        features['starts_with_1'] = int(address.startswith('1'))
        features['starts_with_3'] = int(address.startswith('3'))
        features['starts_with_bc1'] = int(address.startswith('bc1'))
        features['starts_with_T'] = int(address.startswith('T'))

        # Entropy
        probabilities = [count / len(address) for count in char_counts.values()]
        features['entropy'] = -sum(p * np.log2(p) for p in probabilities)

        # Regex patterns
        features['is_eth_address'] = int(bool(re.match(r'^0x[a-fA-F0-9]{40}$', address)))
        features['is_btc_address'] = int(bool(re.match(r'^[13][a-km-zA-HJ-NP-Z1-9]{25,34}$|^bc1[ac-hj-np-z02-9]{39,59}$', address)))
        features['is_trx_address'] = int(bool(re.match(r'^T[A-Za-z1-9]{33}$', address)))
        features['is_sol_address'] = int(bool(re.match(r'^[1-9A-HJ-NP-Za-km-z]{32,44}$', address)))

        # N-gram analysis (bigrams)
        bigrams = [address[i:i+2] for i in range(len(address)-1)]
        bigram_counts = Counter(bigrams)
        features['most_common_bigram_count'] = bigram_counts.most_common(1)[0][1] if bigram_counts else 0

        # Statistical features
        ascii_values = [ord(c) for c in address]
        features['mean_ascii'] = np.mean(ascii_values)
        features['median_ascii'] = np.median(ascii_values)
        features['ascii_range'] = max(ascii_values) - min(ascii_values)
        # Positional feature
        features['first_digit_position'] = next((i for i, c in enumerate(address) if c.isdigit()), -1)

        #checksum binary value features to differentiate b/w eth,bsc and avax
        features['eth_bsc_checksum_valid'] = self.eth_bsc_checksum_valid(address)
        features['avax_checksum_valid'] = self.avax_checksum_valid(address)

        return features


def train_and_save_model(input_csv, model_output_path):
    # Load the dataset
    df = pd.read_csv(input_csv)

    # Split the data
    X = df['Account']
    y = df['Blockchain']

    # Use LabelEncoder for the target variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Initial split into train+validation and holdout sets
    X_train_val, X_holdout, y_train_val, y_holdout = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Create the pipeline
    pipeline = Pipeline([
        ('feature_extractor', CryptoAddressFeatureExtractor()),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True))
    ])

    # Perform cross-validation on train+validation set
    cv_scores = cross_val_score(pipeline, X_train_val, y_train_val, cv=5, scoring='accuracy')
    print("Cross-validation scores:", cv_scores)
    print("Mean CV score:", cv_scores.mean())
    print("Standard deviation of CV score:", cv_scores.std())

    # Train the model on the entire train+validation set
    pipeline.fit(X_train_val, y_train_val)

    # Get OOB error for each tree
    estimator = pipeline.named_steps['classifier']
    error_rate = []
    for i in range(len(estimator.estimators_)):
        estimator.set_params(n_estimators=i+1)
        estimator.fit(pipeline.named_steps['feature_extractor'].transform(X_train_val), y_train_val)
        oob_error = 1 - estimator.oob_score_
        error_rate.append(oob_error)

    # Plot the OOB error
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(estimator.estimators_) + 1), error_rate, color='blue', linestyle='-', marker='o')
    plt.xlabel('Number of Trees')
    plt.ylabel('OOB Error Rate')
    plt.title('OOB Error Rate vs Number of Trees')
    plt.tight_layout()
    plt.savefig('oob_error_plot.png')
    plt.close()

    # Evaluate on the holdout set
    y_holdout_pred = pipeline.predict(X_holdout)
    print("\nHoldout Set Performance:")
    print(classification_report(y_holdout, y_holdout_pred, target_names=le.classes_))

    # Save the model and label encoder
    joblib.dump(pipeline, model_output_path)
    joblib.dump(le, 'label_encoder.joblib')

    print(f"\nModel has been saved to {model_output_path}")
    print("OOB error plot has been saved as 'oob_error_plot.png'")

def predict_blockchain(address, model_path):
    # Load the model and label encoder
    pipeline = joblib.load(model_path)
    le = joblib.load('label_encoder.joblib')

    prediction = pipeline.predict([address])[0]
    return le.inverse_transform([prediction])[0]

def predict_blockchain_batch(input_file, model_path, output_file):
    # Load the model and label encoder
    pipeline = joblib.load(model_path)
    le = joblib.load('label_encoder.joblib')

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        outfile.write("Address,Predicted_Blockchain\n")
        for address in infile:
            address = address.strip()
            if address:
                prediction = pipeline.predict([address])[0]
                predicted_blockchain = le.inverse_transform([prediction])[0]
                outfile.write(f"{address},{predicted_blockchain}\n")

    print(f"Batch predictions have been saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Train model or predict blockchain for an address")
    parser.add_argument("action", choices=["train", "predict", "predict_batch"], help="Choose to train the model or predict for an address")
    parser.add_argument("--input", default="datasets/address_dataset_shuffled.csv",help="Input CSV file for training or address for prediction")
    parser.add_argument("--model", default="crypto_classifier_model.joblib", help="Path to save/load the model")
    parser.add_argument("--output", help="Output file for batch predictions")

    args = parser.parse_args()

    if args.action == "train":
        if not args.input:
            parser.error("--input is required for training (path to input CSV)")
        train_and_save_model(args.input, args.model)
    elif args.action == "predict":
        if not args.input:
            parser.error("--input is required for prediction (address to predict)")
        predicted_blockchain = predict_blockchain(args.input, args.model)
        print(f"The predicted blockchain for {args.input} is: {predicted_blockchain}")
    elif args.action == "predict_batch":
        if not args.input:
            parser.error("--input is required for batch prediction (path to input text file)")
        if not args.output:
            parser.error("--output is required for batch prediction (path to output CSV file)")
        predict_blockchain_batch(args.input, args.model, args.output)

if __name__ == "__main__":
    main()