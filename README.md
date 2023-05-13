
# Bike Price Predictor

A simple flask web application powered by xgboost that helps in predicting the prices of the bike based on given inputs



## Project Highlights
- model accuracy of 92.5% has been achieved
- Generalised model with train accuracy of 96% and test of 92.5%
- Detailed jupyter notebook with each and every step explanined
- Usage of sklearn pipelines [ for training 11 different models synchronously ]



## Technologies Used
- Python
- Flask
- Pandas
- Numpy
- Seaborn
- Scikit-learn
- Html
- Css
- Pickle
## Dataset Description
Dataset is taken from kaggle and can be downloaded from [here](https://www.kaggle.com/datasets/ropali/used-bike-price-in-india)

This contains information about different bikes and their prices and has 7857 rows and 8 columns

| Column name      | Description                                                |
| :----------------| :----------------------------------------------------------|
| model_name          | The name of the bike's model. It contains some additional information like model year,engine etc.                        |
| model_year         | The year in which the model was built.   |
| kms_driven         | Total kilometers the bike has been driven.                             |
| owner | The represents which type of owner the bike has like it is first owner which means the current owner had bought the this bike as new, second owner means the bike has been sold to this owner from first owner and so on.                |
| location              | The location of the seller.                                |
| mileage              | Average mileage the bike gives. Its is represented as kilometer per liter of petrol (kmpl).                                  |
| power           | Power is in terms of Bhp. BHP is the rate at which the torque generated by the engine in a bike is delivered to the wheels. Such that faster the deliverability, higher is the speed of the motorcycle and vice versa. For a bike that consists of a lower BHP can pull higher loads and for a bike that contains a greater BHP can propel the bike at faster speeds.|






## Run Locally

Clone the project

```bash
git clone https://github.com/RishiBakshii/Bike-Price-Predictor.git
```

Go to the project directory

```bash
cd path/to/the/cloned/repository
```

Install dependencies

```bash
pip install -r requirements.txt
```

Start the server

```bash
py app.py
```


## Lifecycle of this Project
- Data Cleaning and Pre-Processing
- Exploratory Data analysis
- Feature Engineering
- Modelling
- Deployment


## Data Cleaning and Pre-Processing
- Columns like model_name, mileage ,kms_driven and power were like this in the initial stage![](https://github.com/RishiBakshii/Bike-Price-Predictor/blob/main/static/images/data_cleaning.png?raw=true)

- A lot of data Cleaning has been perfomed to clean them and make it look like this, all the cleaning functions were written from scratch
- all the values in different units like HP and Kw has been converted to bhp in the power column![](https://github.com/RishiBakshii/Bike-Price-Predictor/blob/main/static/images/after-cleaning.png?raw=true)
## Exploratory Data Analysis
- #### ***Royal enfield*** is the highest selling bike with the ***selling rate of 23.40%***
- #### then comes **Bajaj Pulsar** at **14.78%** selling rate
- #### ***TVS Apache*** is at **6.48%**
- #### ***Bajaj Avenger*** is at **5.21%**
- #### ***Yamaha YZF-R15*** at ***3.84%***
- #### ***harley davidon fat*** is the most expensive bike in the dataset which average price goes up to 9.85 lakhs almost 1 Crore
- #### most of the bikes are manufactured between 2010 and 2020 
- #### Almost all of the bikes are driven under 2 lakh Kilometres
- #### Delhi is the main base of bike buisness ( sales )
- #### as delhi contributes 21.67% to the total sales
- #### mumbai contributes 12% and bangalore contributes 11% in the sales of bike
- #### Ranchi has the highest avg of bike price i.e 1.70Lakhs
- #### Power and Price have have a very strong relationship
- #### Distribution of the target column price is extremely right skewed

## Feature Engineering
- Treatment of Outliers in target column  "Price"![](https://github.com/RishiBakshii/Bike-Price-Predictor/blob/main/static/images/outlier_treatment_price.png?raw=true)

- Fixed the skewed Distribution of Price![](https://github.com/RishiBakshii/Bike-Price-Predictor/blob/main/static/images/skewed_dis_fix.png?raw=true)

- Created column transformers for encoding( onehotencoding and Ordinal Encoding ) and scaling ( MinMaxScaler ) of data
```python
 encoder_transformer=ColumnTransformer([
    ('onehotencoding',OneHotEncoder(sparse=False,handle_unknown='ignore',drop='first'),[0,4]),
    ('ordinalencoder',OrdinalEncoder(categories=[['fourth owner or more','third owner','second owner','first owner']],handle_unknown='error'),[3]),
],remainder='passthrough')

scaler_transformer=ColumnTransformer([
    ('StandardScaler',MinMaxScaler(),[1,2,5,6]),
],remainder='passthrough')
```

## Modelling
- Created piplines for iterative training of 11 different models
```python
pipeline_lr=Pipeline([('encoder_transfomer',encoder_transformer),
                        ('scaler_transformer',scaler_transformer),
                      ('linear',LinearRegression())
                      ])

pipeline_las=Pipeline([('encoder_transformer',encoder_transformer),
                        ('scaler_transformer',scaler_transformer),
                     ('lasso',Lasso())
                     ])

pipeline_ridge=Pipeline([('encoder_transformer',encoder_transformer),
                        ('scaler_transformer',scaler_transformer),
                     ('ridge',Ridge())
                     ])

pipeline_knn=Pipeline([('encoder_transformer',encoder_transformer),
                        ('scaler_transformer',scaler_transformer),
                     ('knn',KNeighborsRegressor())
                     ])

pipeline_dt=Pipeline([('encoder_transformer',encoder_transformer),
                        ('scaler_transformer',scaler_transformer),
                     ('dt',DecisionTreeRegressor())
                     ])

pipeline_svm=Pipeline([('encoder_transformer',encoder_transformer),
                        ('scaler_transformer',scaler_transformer),
                     ('svm',SVR())
                     ])

pipeline_rf=Pipeline([('encoder_transformer',encoder_transformer),
                        ('scaler_transformer',scaler_transformer),
                      ('rf',RandomForestRegressor())
                      ])

pipeline_gbr=Pipeline([('encoder_transformer',encoder_transformer),
                        ('scaler_transformer',scaler_transformer),
                     ('gbr',GradientBoostingRegressor())
                     ])

pipeline_abr=Pipeline([('encoder_transformer',encoder_transformer),
                        ('scaler_transformer',scaler_transformer),
                     ('abr',AdaBoostRegressor())
                     ])

pipeline_etr=Pipeline([('encoder_transformer',encoder_transformer),
                        ('scaler_transformer',scaler_transformer),
                     ('etr',ExtraTreesRegressor())
                     ])

pipeline_xgb=Pipeline([('encoder_transformer0',encoder_transformer),
                        ('scaler_transformer',scaler_transformer),
                     ('xgb',XGBRegressor())
                     ])
```
- ### Training of all the pipelines
  - ![](https://github.com/RishiBakshii/Bike-Price-Predictor/blob/main/static/images/model-training.png?raw=true)

- ### Model's Performance![](https://github.com/RishiBakshii/Bike-Price-Predictor/blob/main/static/images/score-of-the-model.png?raw=true)
    XgBoost was the most generalized model with the highest accuracy and lowest differnece between bias and variance

- ## Model Evaluation
  ### Residual plot
  ![](https://github.com/RishiBakshii/Bike-Price-Predictor/blob/main/static/images/residual_plot.png?raw=true)
    - Residuals are densely populated between the range of -100 and 100
    - some outliers are present in the lower magnitude

  ### Relationship between actual and predicted values
  ![](https://github.com/RishiBakshii/Bike-Price-Predictor/blob/main/static/images/relationship-between-predicted-and-actual-values.png?raw=true)
    - There can be seen a very strong linear relationship between the actual and predicted values



## Deployment
- This project is currently deployed at Render
- and can be visited here [Bike Price Predictor](https://bike-price-predictor.onrender.com/)
