from mlflow.tracking import MlflowClient
import mlflow


client = MlflowClient()


run_id = "056ca3c74b0e429d916e6c827a4d5411"

model_path = "file:///C:/Users/MV/Desktop/MLOPS%20-%20Copy%20-%20Copy/model-registry-demo/mlruns/230634692578045324/056ca3c74b0e429d916e6c827a4d5411/artifacts/Random forest"


model_uri = f"runs:/{run_id}/{model_path}"

model_name = "diabetes-rf"


result = mlflow.register_model(model_uri,model_name)


import time
time.sleep(5)



client.update_model_version(
    name=model_name,
    version=result.version,
    description="this is a random forest model trained on pima inidan dataset"
)

client.set_model_version_tag(
     name=model_name,
     version=result.version,
     key="experiment",
     value="diabetes-prediction"
)