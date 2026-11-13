
from kfp import compiler, dsl
from kfp import kubernetes

COMPONENT_IMAGE = "yolo26-kfp:1.0"

WORKSPACE_PVC = "yolo26-workspace"
WORKSPACE_MOUNT = "/workspace"


@dsl.container_component
def run_existing_component(script_name: str):
    return dsl.ContainerSpec(
        image=COMPONENT_IMAGE,
        command=["python"],
        args=[script_name],
    )


@dsl.pipeline(
    name="yolo26-object-detection-pipeline",
    description="YOLO26 object detection training and model registration pipeline",
)
def yolo26_pipeline():

    # 1. Data Preparation
    data_task = run_existing_component(
        script_name="components/data_preparation.py"
    )
    data_task.set_display_name("1 - Data Preparation")
    data_task.set_caching_options(enable_caching=False)

    kubernetes.mount_pvc(
        data_task,
        pvc_name=WORKSPACE_PVC,
        mount_path=WORKSPACE_MOUNT,
    )

    # 2. Model Development
    development_task = run_existing_component(
        script_name="components/model_development.py"
    )
    development_task.set_display_name("2 - Model Development")
    development_task.set_caching_options(enable_caching=False)
    development_task.after(data_task)

    kubernetes.mount_pvc(
        development_task,
        pvc_name=WORKSPACE_PVC,
        mount_path=WORKSPACE_MOUNT,
    )

    # 3. Model Training
    training_task = run_existing_component(
        script_name="components/model_training.py"
    )
    training_task.set_display_name("3 - Model Training")
    training_task.set_caching_options(enable_caching=False)
    training_task.after(development_task)

    kubernetes.mount_pvc(
        training_task,
        pvc_name=WORKSPACE_PVC,
        mount_path=WORKSPACE_MOUNT,
    )

    # 4. Model Optimization
    optimization_task = run_existing_component(
        script_name="components/model_optimization.py"
    )
    optimization_task.set_display_name("4 - Model Optimization")
    optimization_task.set_caching_options(enable_caching=False)
    optimization_task.after(training_task)

    kubernetes.mount_pvc(
        optimization_task,
        pvc_name=WORKSPACE_PVC,
        mount_path=WORKSPACE_MOUNT,
    )

    # 5. Model Registry
    registry_task = run_existing_component(
        script_name="components/model_registry.py"
    )
    registry_task.set_display_name("5 - Model Registry")
    registry_task.set_caching_options(enable_caching=False)
    registry_task.after(optimization_task)

    kubernetes.mount_pvc(
        registry_task,
        pvc_name=WORKSPACE_PVC,
        mount_path=WORKSPACE_MOUNT,
    )


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=yolo26_pipeline,
        package_path="pipeline.yaml",
    )

    print("Pipeline compiled successfully!")
    print("Generated file: pipeline.yaml")
