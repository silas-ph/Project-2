# Project-2

The dataset essentially contains five CSV files:

### stores.csv

This file contains anonymized information about the 45 stores, indicating the type and size of the store.

### train.csv

Historical training data, which covers from 2010-02-05 to 2012-11-01. Within this file, we will find the following fields:

- **Store** - store number
- **Dept** - department number
- **Date** - the week
- **Weekly_Sales** - sales for the given department in the given store
- **IsHoliday** - whether the week is a special holiday week

### features.csv

This file contains additional data related to the store, department, and regional activity for the given dates. It contains the following fields:

- **Store** - the store number
- **Date** - the week
- **Temperature** - average temperature in the region
- **Fuel_Price** - cost of fuel in the region
- **MarkDown1-5** - anonymized data related to promotional markdowns that Walmart is running. MarkDown data is only available after Nov 2011, and is not available for all stores all the time. Any missing value is marked with an NA.
- **CPI** - the consumer price index
- **Unemployment** - the unemployment rate
- **IsHoliday** - whether the week is a special holiday week

### test.csv

This file is identical to train.csv, except we have withheld the weekly sales. You must predict the sales for each triplet of store, department, and date in this file.

---

You can copy and paste this Markdown into any Markdown editor or viewer to see the formatted content.
Linear Regression Model
Linear Regressor Accuracy
Mean Absolute Error
Mean Square Error
Root Mean Squared Error
R2

Random Forest Regression Model

Random Forest Regressor
Mean Absolute Error
Root Mean Squared Error
R2

K Neighbors Regression Model

XGBoost Regression Model
