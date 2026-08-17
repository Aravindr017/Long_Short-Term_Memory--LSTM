# Project: SMS Spam Classification using Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM)

## Overview

This project focuses on building and evaluating Recurrent Neural Network (RNN) and Long Short-Term Memory (LSTM) models to classify SMS messages as either 'spam' or 'ham' (not spam). The goal is to develop an effective text classification system that can distinguish between spam and non-spam messages, exploring the capabilities of different recurrent architectures.

## Dataset

The dataset used for this project consists of SMS messages, each labeled as either 'spam' or 'ham'.

**Source:** (Implicitly from `spam.xlsx`, typically publicly available SMS Spam Collection Dataset)

Each data point includes:

*   **`v1` (class)**: The target variable indicating whether the message is 'ham' or 'spam'.
*   **`v2` (text)**: The actual content of the SMS message.

**Key Characteristics:**
*   **Text Data**: The primary feature for classification is the raw text of the SMS messages.
*   **Binary Classification**: The task involves classifying messages into one of two categories: 'ham' (0) or 'spam' (1).
*   **Imbalanced Dataset**: The dataset is typically imbalanced, with a significantly higher number of 'ham' messages compared to 'spam' messages, which is addressed through stratified splitting.

## Preprocessing & Feature Engineering

To prepare the text data for the RNN and LSTM models, the following preprocessing and feature engineering steps were applied:

1.  **Data Loading and Initial Cleaning**: The dataset was loaded from an Excel file (`spam.xlsx`). Unwanted columns were removed, and the remaining two columns were renamed to `'class'` and `'text'` for clarity.

2.  **Target Encoding**: The categorical 'class' column ('spam'/'ham') was converted into numerical labels: 'spam' mapped to `1` and 'ham' mapped to `0`.

3.  **Data Type Conversion**: The 'text' column was explicitly converted to a string data type to ensure consistent text processing.

4.  **Train-Test Split**: The dataset was split into training and testing sets (80% train, 20% test). `stratify=y` was used to maintain the same proportion of 'spam' and 'ham' messages in both the training and testing sets, addressing the class imbalance.

5.  **Tokenization**: A `Tokenizer` from `tensorflow.keras.preprocessing.text` was used to convert text messages into sequences of integers. Only the 5000 most frequent words were considered (`num_words=5000`), and out-of-vocabulary words were marked with `<OOV>`.

    ```python
    tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)
    train_seq = tokenizer.texts_to_sequences(X_train)
    test_seq = tokenizer.texts_to_sequences(X_test)
    ```

6.  **Padding Sequences**: The integer sequences representing messages were padded to a uniform length (`MAX_LENGTH = 100`) using `pad_sequences`. This ensures all input sequences to the RNN have the same dimension.

    ```python
    MAX_LENGTH = 100
    X_train_pad = pad_sequences(train_seq, MAX_LENGTH, padding='post', truncating='post')
    X_test_pad = pad_sequences(test_seq, MAX_LENGTH, padding='post', truncating='post')
    ```

## Models Used

Two types of Recurrent Neural Networks were constructed using Keras: a SimpleRNN and an LSTM model. Both models are designed to process the padded integer sequences and classify them.

### RNN Model Architecture (`RNN_model`)

**Architecture:**
*   **Embedding Layer**: Converts the integer-encoded words into dense vectors of fixed size. `input_dim=5000` (number of unique words) and `output_dim=64` (dimension of the dense embedding).
*   **SimpleRNN Layer**: A basic recurrent layer with 64 units, capable of processing sequences.
*   **Dense Layer (ReLU)**: A fully connected hidden layer with 32 units and ReLU activation.
*   **Dense Layer (Sigmoid)**: The output layer with 1 unit and sigmoid activation, suitable for binary classification.

    ```python
    RNN_model = Sequential([
        Embedding(input_dim=5000, output_dim=64),
        SimpleRNN(64),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    ```

**Training Details:**
*   **Optimizer**: `adam`
*   **Loss Function**: `binary_crossentropy`
*   **Metrics**: `accuracy`
*   **Epochs**: 10
*   **Batch Size**: 32
*   **Validation Split**: 0.2

### LSTM Model Architecture (`LSTM_model`)

**Architecture:**
*   **Embedding Layer**: Similar to RNN, converts integer-encoded words into dense vectors. `input_dim=5000` and `output_dim=64`.
*   **LSTM Layer**: A Long Short-Term Memory layer with 64 units. LSTMs are designed to overcome the vanishing gradient problem of simple RNNs and are better at capturing long-term dependencies in sequences.
*   **Dense Layer (ReLU)**: A fully connected hidden layer with 32 units and ReLU activation.
*   **Dense Layer (Sigmoid)**: The output layer with 1 unit and sigmoid activation for binary classification.

    ```python
    LSTM_model = Sequential([
        Embedding(input_dim=5000, output_dim=64),
        LSTM(64),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    ```

**Training Details:**
*   **Optimizer**: `adam`
*   **Loss Function**: `binary_crossentropy`
*   **Metrics**: `accuracy`
*   **Epochs**: 10
*   **Batch Size**: 32
*   **Validation Split**: 0.3
*   **Model Checkpointing**: A `ModelCheckpoint` callback is used to save the model (`best_lstm_model.keras`) with the best `val_accuracy` during training.

## Evaluation Metrics

The models' performances were evaluated using:

*   **Accuracy**: The proportion of correctly classified messages on the test set.
*   **Loss**: The `binary_crossentropy` value during training and validation.
*   **Training and Validation Loss/Accuracy Plots**: Visualizations showing the change in loss and accuracy over epochs for both training and validation sets.

## Results and Analysis

### RNN Model Performance:
*   **Test Accuracy**: The RNN model achieved a test accuracy of approximately `0.866`.
*   **Prediction Inference**: When tested with example spam and non-spam messages, the RNN model showed limitations. For instance, an obvious spam message like "URGENT! You won a $1,000 Walmart gift card!" was predicted as non-spam with a low probability (e.g., `0.156`).
*   **Reason for Failure (Inference from notebook)**: This lower-than-expected performance in classifying specific spam messages is attributed to the `Tokenizer` being limited to the 5000 most frequent words. This means that many keywords or phrases crucial for identifying spam, which might not be among the top 5000 most frequent words in the overall dataset, were effectively ignored or mapped to `<OOV>` tokens. Consequently, the model lacked the necessary features to accurately distinguish these specific spam patterns.

### LSTM Model Performance:
*   **Test Accuracy**: The LSTM model also achieved a test accuracy of approximately `0.866`, which is very similar to the SimpleRNN model.
*   **Prediction Inference**: Similar to the RNN, the LSTM model also misclassified the example spam message, predicting it as non-spam with a low probability (e.g., `0.125`).
*   **Analysis**: While LSTMs are generally more powerful for sequence processing due to their ability to capture long-term dependencies, the similar performance to the SimpleRNN model in this specific case, and the continued misclassification of obvious spam, suggests that the primary bottleneck might not be the model architecture itself, but rather the initial data preprocessing, particularly the limited vocabulary size and potentially the way the training data was fed to the model (e.g., training with `X_test_pad` for `y_train` could lead to unexpected behavior).

## Potential Improvements

To enhance the model's performance and address the identified limitations, several strategies can be employed:

1.  **Increase Vocabulary Size**: Expand the `num_words` parameter in the `Tokenizer` to include a larger set of words. This would allow the model to capture more diverse and potentially spam-specific keywords that are currently being filtered out.

2.  **Refine Text Preprocessing**: Implement more sophisticated text cleaning techniques:
    *   **Custom Stop Word Handling**: Evaluate if removing common stop words (or defining a custom list) could improve signal-to-noise ratio, especially for spam detection where common words might not carry much predictive power.
    *   **Punctuation and Special Character Normalization**: Spam messages often use unusual punctuation, symbols, or URLs. Better strategies for handling or encoding these features could be beneficial.
    *   **Stemming/Lemmatization**: Reduce words to their base forms to group variations of the same word, which can help in reducing the vocabulary size while retaining meaning.
    *   **URL/Number Handling**: Replace URLs or numbers with generic tokens (e.g., `<URL>`, `<NUMBER>`) to reduce vocabulary size and capture patterns related to their presence.

3.  **Hyperparameter Tuning**: Systematically optimize various hyperparameters:
    *   **`MAX_LENGTH`**: Experiment with different sequence lengths to find the optimal balance between capturing context and computational cost.
    *   **Embedding Dimension (`output_dim`)**: Tune the size of the word embeddings.
    *   **RNN/LSTM Units**: Adjust the number of units in the recurrent layers.
    *   **Learning Rate and Batch Size**: Optimize these for faster and more stable training.

4.  **Regularization Techniques**: Introduce `Dropout` layers within the RNN/LSTM architecture to prevent overfitting, especially with larger models or datasets.

5.  **More Advanced Model Architectures**: Explore more complex recurrent architectures:
    *   **Bidirectional LSTMs (BiLSTMs)**: Process sequences in both forward and backward directions, allowing the model to capture context from both past and future words.
    *   **Stacked LSTMs**: Use multiple LSTM layers stacked on top of each other to learn hierarchical representations.
    *   **Transformer-based Models**: For highly complex language understanding tasks, models like BERT or GPT could offer superior performance, although they are more computationally intensive.

6.  **Addressing Data Imbalance**: While stratified sampling helps in splitting, further techniques such as oversampling (e.g., SMOTE) or undersampling could be considered to balance the classes in the training data, potentially improving the model's ability to learn spam patterns.

7.  **Correct Data Usage**: Ensure that the training data (`X_train_pad`) is consistently used for model fitting, and validation/test data is used appropriately for evaluation.

## How to Run the Project

### Prerequisites

*   Python 3.x
*   `pandas`
*   `numpy`
*   `scikit-learn`
*   `tensorflow` (including `keras`)
*   `matplotlib`
*   `seaborn`
*   `openpyxl` (for reading `.xlsx` files)

### Installation

Install the necessary libraries using pip:

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn openpyxl
```

### Code Structure (Conceptual Steps)

1.  **Import Libraries**: Import all necessary libraries for data handling, preprocessing, model building, and visualization.
2.  **Load Dataset**: Load the `spam.xlsx` file into a pandas DataFrame.
3.  **EDA (Exploratory Data Analysis)**: Inspect data types, value counts, and the head of the DataFrame.
4.  **Preprocessing**: Clean data by selecting relevant columns, renaming, encoding the target, converting text to string, and splitting into train/test sets.
5.  **Word Preprocessing**: Initialize `Tokenizer`, fit on training data, convert text to sequences, and pad sequences to a fixed length.
6.  **RNN Model Building**: Define and compile the SimpleRNN model using `Sequential`, `Embedding`, `SimpleRNN`, and `Dense` layers.
7.  **RNN Model Training**: Fit the SimpleRNN model to the training data with specified epochs, batch size, and validation split.
8.  **RNN Model Evaluation**: Predict on the test set and calculate `accuracy_score`.
9.  **LSTM Model Building**: Define and compile the LSTM model using `Sequential`, `Embedding`, `LSTM`, and `Dense` layers.
10. **LSTM Model Training**: Fit the LSTM model to the training data, incorporating model checkpointing.
11. **LSTM Model Evaluation**: Predict on the test set and calculate `accuracy_score`.
12. **Visualize Results**: Plot training/validation loss and accuracy over epochs for both models using `matplotlib`.
13. **Prediction Functions**: Implement functions for making predictions on new text messages using both the trained RNN and LSTM models.

Refer to the notebook for the full implementation details and specific code for each step.
