# Exploratory Data Analysis (EDA) using Python

## Introduction 

### Exploratory Data Analysis (EDA) 
**Exploratory Data Analysis (EDA)** is a method of analyzing datasets to understand their main characteristics, beyond the formal modeling and thereby contrasts traditional hypothesis testing. 
It involves summarizing data features, detecting patterns, and uncovering relationships through visual and 
statistical techniques. 

### Data Pre-processing and Feature Engineering
**Data pre-processing** involves filtration and preparing raw data before it is analyzed to facilitate deature engineering. 

**Feature Engineering** refers to the process of using domain knowledge to selecta and transform the most relevant variables from raw data when creating a predictive model using machine learning or statistical modeling. The goal of feature engineering and selection is to improve the performance of machine learning algorithms.

![image](https://github.com/JaimeRosique/ComputerScience/assets/118359274/a88195a7-eadb-4350-812f-f1ed234794ca)

**Processes**
- **Feature Creation:** Process that involves transforming, aggregating, or creating new features to enhance the predictive power of models. The goal is to make the data more informative and suitable for the algorithms. Techniques can include creating interaction terms, scaling, encoding categorical variables, or deriving new metrics.
- **Transformations:** Process that involves manipulating the predictor variables to imporve model performance; e.g. ensuring the model is flexible in the variety of data it can ingest; ensuring variables are the same scale, making the model easier to understand; improving accuracy; and avoiding computational errors by ensuring all features are within an acceptable range for the model.
- **Feature Extraction:** Process that transforms raw data into a set of meaningfull features that can be used for model building. The goal is to reduce the amount of data by selecting key attributes that represent the underlying structure of the data.
- **Feature Selection:** Process where a subset of relevant features is chosen from a larger set of available features. The goal is to improve the performnace of a model by reducing overfitting, improving accuracy, and decreasing training time.

## Step 1: Import Python Libraries

Import libraries required for the analysis such as:

- **Pandas:** For data manipulation and analysis
  ```python
  import panda as pd
- **NumPy:** For numerical operations
  ```python
  import numpy as np
- **Matplotlib:** For basic plotting
   ```python
  import matplotlib.pyplot as plt
- **Seaborn:** For statistical data visualization
   ```python
  import seaborn as sns
- **SciPy:** For statistical and scientific computations
   ```python
  import scipy
- **Plotly:** For interactive visualizations (optional)
   ```python
  import plotly.express as px

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import plotly.express as px
# to ignore warnings
import warnings
warnings.filterwarnings('ignore')
```

## Step 2: Load Dataset

In order to load a dataset, 'pandas' is an excellent choice due to its optimization and poweful data manipulation capabilities. Offering simple and intuitive functions for reading various data formats, such as CSV, Excel and SQL. This versatility, combined whith its user-friendly interface, makes 'pandas' a preferred tool for data analysis tasks.

Since most of the data are available in tabular format of CSV files, converting this data  into a pandas DataFrame it straightforward using the **read_csv()**. The 'csv' can be replaced whith other data formats as needed.

```python
# For a CSV file
data = pd.read_csv('path/to/your/dataset.csv')

# For an Excel file
data = pd.read_excel('path/to/your/dataset.xlsx')

# For a SQL database
from sqlalchemy import create_engine
engine = create_engine('sqlite:///path/to/your/database.db')
data = pd.read_sql('SELECT * FROM your_table', engine)
```
For more information on using pandas, 
[refer to its official documentation](https://pandas.pydata.org/docs/)

### Understand the structure of the dataset

The main goal of data understanding is to gain further insight about the data.

- Getting the dimensions of the dataset (**data.shape**):
  - Returns a tuple that shows the dimensions of the dataset, where the first element is the number of rows and the second is the number of columns. This gives an overview of how much data is available and the structure of the dataset
  - ```python
    data.shape
    ```  
- Getting the first 5 rows of the dataset (**data.head()**):
```python
data.head()
```
- Getting the last 5 rows of the dataset (**data.tail()**):
```python
data.tail()
```
- Getting a summary of the dataset (**data.info()**):
  -  Provides a concise summary of the dataset 'data', inluding the number of non-null values in each column, the data types of each column, and the memory usage. This helps in understanding the completeness of the dataset and the types of data.
```python
data.info()
```
- Getting a summary statistics (**data.describe()**):
  - Generates descriptive statistics that summarize the central tendency, dispersion and shape of the datasets distribution for numerical columns. Including statistics such as count, mean, standard deviation, minimum, maximum and quartile values.
```python
data.describe()
```
- Getting the numbes of unique values in each column (**data.unique()**):
```python
data.nunique()
```
- Getting the number of null values in each column (**data.isnull().sum()**):
```python
data.nunique()
#To calculate the percentage of missing values
(data.isnull().sum()/(len(data)))*100
```  
## Step 3: Feature Engeneering
Process of using domain knowledge to create new features from raw data that can enhance the performance of machine learning models, creating meaningful data from raw data.

### Data reduction
Eliminate columns or rows:
```python
#Remove a colum from data
data = data.drop(['column_name'], axis = 1)
#axis = 1: meanes columns
#axis = 0: means rows
```

### Creating new features
Combining or transforming existing data into new variables that have greater predictive power.
```python
data['new_variable'] = "new_data"
#example
data = {
  'Height_cm': [150,160,170],
  'Weight_cm': [50, 60, 70]
}
data = pd.DataFrame(data)

data['Height_m'] = data['Height_cm'] / 100
data['BMI'] = data['Weight_kg'] / data['Height_m'] ** 2)
data['Name'] = data.DNI.str.split().str.get(0)
data[['Name', 'BMI', 'Height_m']]
```
### Selecting Features
Choosing the most relevant features for model building to improve accuracy and reduce complexity.
Selecting the right features is a crucial step, here are some common techniques to identify important features:

#### Correlation matrix
Is a table showing correlation coefficients between variables. Each cell in the table shows the correlation between two variables. The value ranges from 0 to 1, showing the interdependency in relations asociated or between each pair of variables and every at the same time.
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sample DataFrame
data = {'A': [1, 2, 3], 'B': [2, 4, 5], 'C': [5, 6, 7]}
df = pd.DataFrame(data)

# Compute correlation matrix
corr_matrix = df.corr()

# Visualize correlation matrix
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()
```
![image](https://github.com/JaimeRosique/ComputerScience/assets/118359274/dafb1639-a8a9-4b85-bdb6-693f8db44c23)

#### Feature Importance 
