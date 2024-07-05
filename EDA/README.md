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

### Step 1: Import Python Libraries

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
