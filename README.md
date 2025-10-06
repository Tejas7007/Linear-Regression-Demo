# Linear Regression on the Boston Housing Dataset

This repository demonstrates an end-to-end Linear Regression project built in Python using the Boston Housing dataset. The objective is to predict the median value of owner-occupied homes (MEDV) given various socio-economic, demographic, and housing-related features. The project includes dataset exploration, regression modeling, evaluation, and visualization to understand how these features influence housing prices.

## Dataset

The Boston Housing dataset is a well-known dataset in machine learning, often used as a benchmark for regression tasks. It contains 506 observations and 14 variables (13 input features and 1 target variable).

- Target variable:
  - `MEDV`: Median value of owner-occupied homes (in $1000s).

- Input features:

| Feature   | Description |
|-----------|-------------|
| CRIM      | Per capita crime rate by town |
| ZN        | Proportion of residential land zoned for lots over 25,000 sq.ft. |
| INDUS     | Proportion of non-retail business acres per town |
| CHAS      | Charles River dummy variable (1 if tract bounds river; 0 otherwise) |
| NOX       | Nitric oxide concentration (parts per 10 million) |
| RM        | Average number of rooms per dwelling |
| AGE       | Proportion of owner-occupied units built before 1940 |
| DIS       | Weighted distance to employment centers |
| RAD       | Index of accessibility to radial highways |
| TAX       | Full-value property-tax rate per $10,000 |
| PTRATIO   | Pupil-teacher ratio by town |
| B         | 1000(Bk - 0.63)² where Bk is the proportion of Black residents |
| LSTAT     | % lower status of the population |

## Workflow

1. **Data Loading**  
   Loaded the dataset using pandas and inspected its structure with `.head()`, `.info()`, and `.describe()`.

2. **Exploratory Data Analysis (EDA)**  
   - Checked for missing values.  
   - Visualized feature distributions and relationships with MEDV.  
   - Built a correlation heatmap to identify the most significant predictors.

3. **Data Preprocessing**  
   - Defined features (X) as the 13 independent variables and target (y) as MEDV.  
   - Split dataset into training (80%) and testing (20%) sets.  

4. **Model Training**  
   - Applied `LinearRegression` from scikit-learn.  
   - Trained the model on the training set and extracted intercepts and coefficients.  

5. **Evaluation**  
   - Predictions were made on the test set.  
   - Model performance was measured with:  
     - Mean Absolute Error (MAE)  
     - Mean Squared Error (MSE)  
     - Root Mean Squared Error (RMSE)  
     - R² Score  

6. **Visualization**  
   - Predicted vs actual plot to show model accuracy.  
   - Residual plots to analyze model errors.  

## Results

- Mean Absolute Error (MAE): ~3.1  
- Mean Squared Error (MSE): ~21.5  
- Root Mean Squared Error (RMSE): ~4.6  
- R² Score: ~0.72  

The model explains about 72% of the variance in housing prices. While results are reasonable, there is scope for improvement with more advanced techniques.

## Future Work

- Apply regularization methods (Ridge, Lasso) to reduce multicollinearity.  
- Experiment with Polynomial Regression or tree-based models.  
- Add cross-validation for robust evaluation.  
- Deploy the model as a small web application.  

## Tools and Libraries

- Python  
- pandas, numpy  
- matplotlib, seaborn  
- scikit-learn  


