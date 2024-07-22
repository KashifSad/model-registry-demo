from mlflow.tracking import MlflowClient


client = MlflowClient()


model_name = "diabetes-rf"
model_version = 2


new_stage = "Production"

client.transition_model_version_stage(
    name=model_name,
    version=model_version,
    stage=new_stage,
    archive_existing_versions=True
)


print(f"MODdel version {model_version} transited to {new_stage}")