import csv
import random
import argparse
import os

def process_csv(input_filename, output_filename, account_key, blockchain):
    with open(input_filename, mode='r', newline='') as infile, \
         open(output_filename, mode='w', newline='') as outfile:
        reader = csv.DictReader(infile)
        writer = csv.writer(outfile)
        writer.writerow(['Account', 'Blockchain'])
        for row in reader:
            writer.writerow([row[account_key], blockchain])

def combine_and_shuffle_csv_files(input_files, combined_output, shuffled_output):
    # Combine CSV files
    with open(combined_output, mode='w', newline='') as outfile:
        writer = csv.writer(outfile)
        for i, filename in enumerate(input_files):
            with open(filename, mode='r', newline='') as infile:
                reader = csv.reader(infile)
                if i == 0:
                    writer.writerow(next(reader))  # Write the header
                else:
                    next(reader)
                writer.writerows(reader)
    
    print(f"Data has been combined into {combined_output}")

    # Shuffle the combined CSV file
    with open(combined_output, mode='r', newline='') as infile:
        reader = csv.reader(infile)
        header = next(reader)
        rows = list(reader)

    random.shuffle(rows)

    with open(shuffled_output, mode='w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Data has been shuffled and written to {shuffled_output}")

def main():
    parser = argparse.ArgumentParser(description="Process and combine CSV files for blockchain addresses.")
    parser.add_argument("--eth_input", default="datasets/eth_addresses.csv", help="Input file for Ethereum addresses")
    parser.add_argument("--btc_input", default="datasets/labels_transactionsagg.csv", help="Input file for Bitcoin addresses")
    parser.add_argument("--output_dir", default="datasets", help="Output directory for processed files")
    args = parser.parse_args()

    # Process ETH and BTC CSV files
    process_csv(args.eth_input, os.path.join(args.output_dir, "eth.csv"), 'Address', 'ethereum')
    process_csv(args.btc_input, os.path.join(args.output_dir, "btc.csv"), 'address', 'bitcoin')

    # List of CSV files to combine
    csv_files = [
        os.path.join(args.output_dir, file) for file in 
        ['btc.csv', 'eth.csv', 'sol.csv', 'trx.csv', 'bsc_out.csv', 'polygon_out.csv', 'avax_out.csv']
    ]

    combined_csv = os.path.join(args.output_dir, 'combined_dataset.csv')
    shuffled_csv = os.path.join(args.output_dir, 'address_dataset_shuffled.csv')

    combine_and_shuffle_csv_files(csv_files, combined_csv, shuffled_csv)

if __name__ == "__main__":
    main()