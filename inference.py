import numpy as np
import mlflow.pyfunc



data = np.array([3,7,38,56,38,67,49,3,32]).reshape(1,-1)


model_name = "diabetes-rf"
version = "2"


model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{version}")


print(model.predict(data))