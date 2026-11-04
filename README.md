# Fraud-Detection-With-Logistic-Regression
A short scala script I created for a classwork assignment that uses a logistic regression model to determine whether a transaction is fraudulent.<br><br>

**Objective:** Implement a logistic regression model to identify potentially fraudulent online transactions.<br><br>

**Data Description:**
 - **Columns:**
     - transaction_id: The identifier for a transaction. String.
     - transaction_amount: The amount tendered for a transaction. Double.
     - transaction_frequency: How many times the transaction occurred. Int.
     - label: A binary column representing whether the transaction is fraudulent (0 = False, 1 = True). Int.     
     - **Row Count:** 1200 entries
<br>

**Data Pipeline:**
1. Load data: Read logistic_fraud.csv into a Spark DataFrame with headers and schema inference.
2. Clean: Drop rows where every column is null.
3. Assemble features: Use VectorAssembler to combine transaction_amount, transaction_frequency, and location_risk_score into a single features vector.
4. Split: Train on the first half of the data (transaction_id <= 'TXN00601') and test on the second half (transaction_id > 'TXN00601').
5. Train: Fit a LogisticRegression model on the training set using the features and label columns.
6. Predict and evaluate: Score the test set at a 0.5 threshold and compute accuracy with MulticlassClassificationEvaluator.
7. Threshold comparison: Change the model's threshold to 0.8, re-score the same test set, and compare accuracy.
<br>

**Usage:**
<br>
Place logistic_fraud.csv where the script expects it, or update the path in spark.read...csv(...).
Run the script in spark-shell or as a Spark application.
<br>

**OutputL**
- The original data and the test data with assembled features
- Test-set predictions with the actual label and predicted label for each transaction
- Accuracy at the 0.5 threshold
- Accuracy at the 0.8 threshold
