# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # ![alt text](https://rakirahman.blob.core.windows.net/public/images/Misc/Flask.png "Microsoft Intelligent Cloud Global Blackbelt") &nbsp;MLFlow 101
# MAGIC > with Automatic MLflow Logging
# MAGIC 
# MAGIC ### Setup: Imports and Load Data
# MAGIC To keep the focus on the toolset, we use the simple, built in _Iris dataset_ with _Scikit-learn_ - but has built in integration for a large number of libraries.
# MAGIC > E.g. TesorFlow and Keras, Gluon, XGBoost, LightGBM, Statsmodels, Spark, Fastai, Pytorch ... (and Custom Libraries)

# COMMAND ----------

from sklearn import datasets, linear_model, tree
import pandas as pd
iris = datasets.load_iris()
print("Feature Data: \n", iris.data[::50], "\nTarget Classes: \n", iris.target[::50])

# COMMAND ----------

# MAGIC %md
# MAGIC 💡 **Note:** At this point - the idea is feature engineering would have already been completed (this notebook or earlier), and we're ready to begin experimenting.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model 1: LogisticRegression
# MAGIC 
# MAGIC We run a simple Logistic Regression model:

# COMMAND ----------

model_1 = linear_model.LogisticRegression(max_iter=200)
model_1.fit(iris.data, iris.target)

# COMMAND ----------

# MAGIC %md
# MAGIC And immediately see all the **parameters** (default in this case) and **metrics** that were logged as part of this run on the right 👉

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model 2: Decision Tree
# MAGIC 
# MAGIC We train another model - to showcase this once more.

# COMMAND ----------

model_2 = tree.DecisionTreeClassifier()
model_2.fit(iris.data, iris.target)

# COMMAND ----------

# MAGIC %md
# MAGIC The parameters and metrics tracked are unique to the model (and the library), and we can immediately see which model performed better. ◀

# COMMAND ----------

# MAGIC %md
# MAGIC ### Alternative: Enable MLflow Autologging manually
# MAGIC Above, we saw Autologging in action by default (I had turned it on) - but we can `import mlflow` and manualy perform this as well.

# COMMAND ----------

import mlflow # Import MLflow 
mlflow.autolog() # Turn on "autologging"

with mlflow.start_run(run_name="Sklearn Decision Tree"): # Pass in run_name using "with" Python syntax
  model_3 = tree.DecisionTreeClassifier(max_depth=5).fit(iris.data, iris.target) #Instantiate and fit model

# COMMAND ----------

# MAGIC %md
# MAGIC So far we've trained three simple models, and have the parameters and metrics available for us to compare.
# MAGIC 
# MAGIC To dive in a bit more, we go into the viewer now.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Make Predictions with Model
# MAGIC After registering your model to the model registry and transitioning it to Stage `Production`, load it back to make predictions

# COMMAND ----------

model_name = "mlflow_101_demo" #Or replace with your model name
model_uri = "models:/{}/production".format(model_name)

print("Loading PRODUCTION model stage with name: '{}'".format(model_uri))
model = mlflow.pyfunc.load_model(model_uri)
print("Model object of type:", type(model))

# COMMAND ----------

predictions = model.predict(pd.DataFrame(iris.data[::50]))
pd.DataFrame(predictions).head()
