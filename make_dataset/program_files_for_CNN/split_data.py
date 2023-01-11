import pandas as pd


def split_data(path):

    print(f"path: {path}")
    # Load the CSV file into a pandas DataFrame, keeping the headers
    df = pd.read_csv(f"{path}//dataset.csv", header=0)

    # Shuffle the rows
    df = df.sample(frac=1)

    # Split the data into a training set (80%) and a validation set (20%)
    training_set_size = int(len(df) * 0.8)
    training_set = df[:training_set_size]
    validation_set = df[training_set_size:]

    # Save the training and validation sets to separate CSV files, including the headers
    training_set.to_csv(f"{path}/training_set.csv", index=False, header=True)
    validation_set.to_csv(f"{path}/validation_set.csv", index=False, header=True)

if __name__ == '__main__':
    split_data()