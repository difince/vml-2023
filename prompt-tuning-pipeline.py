import kfp.dsl as dsl
from kfp.dsl import component
from kfp.compiler import Compiler
import kfp.components as comp


peft_model_server_image="quay.io/aipipeline/peft-model-server:latest"
modelmesh_namespace="modelmesh-serving"
modelmesh_servicename="modelmesh-serving"
pipeline_out_file="llm-prompt_tuning_pipeline.yaml"
kserv_component="https://raw.githubusercontent.com/kubeflow/pipelines/release-2.0.1/components/kserve/component.yaml"

@component(
    packages_to_install=["peft", "transformers", "datasets", "torch", "datasets", "tqdm"],
    base_image='python:3.10'
)
def prompt_tuning_bloom(peft_model_publish_id: str, model_name_or_path: str, num_epochs: int, hf_token: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator, get_linear_schedule_with_warmup
    from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType
    import torch
    from datasets import load_dataset
    import os
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import base64

    peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.TEXT,
        num_virtual_tokens=8,
        prompt_tuning_init_text="Classify if the tweet is a complaint or not:",
        tokenizer_name_or_path=model_name_or_path,
    )

    dataset_name = "twitter_complaints"
    text_column = "Tweet text"
    label_column = "text_label"
    max_length = 64
    lr = 3e-2
    batch_size = 8

    dataset = load_dataset("ought/raft", dataset_name)
    dataset["train"][0]

    classes = [k.replace("_", " ") for k in dataset["train"].features["Label"].names]
    dataset = dataset.map(
        lambda x: {"text_label": [classes[label] for label in x["Label"]]},
        batched=True,
        num_proc=1,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def preprocess_function(examples):
        batch_size = len(examples[text_column])
        inputs = [f"{text_column} : {x} Label : " for x in examples[text_column]]
        targets = [str(x) for x in examples[label_column]]
        model_inputs = tokenizer(inputs)
        labels = tokenizer(targets)
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i] + [tokenizer.pad_token_id]
            model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
            labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i]
            model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
                max_length - len(sample_input_ids)
            ) + sample_input_ids
            model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
                "attention_mask"
            ][i]
            labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
            labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    processed_datasets = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=1,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["train"]


    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=False
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=False)

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    model = get_peft_model(model, peft_config)
    print(model.print_trainable_parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            batch = {k: v for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        model.eval()
        eval_loss = 0
        eval_preds = []
        for step, batch in enumerate(tqdm(eval_dataloader)):
            batch = {k: v for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            eval_loss += loss.detach().float()
            eval_preds.extend(
                tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
            )

        eval_epoch_loss = eval_loss / len(eval_dataloader)
        eval_ppl = torch.exp(eval_epoch_loss)
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        print("epoch=%s: train_ppl=%s train_epoch_loss=%s eval_ppl=%s eval_epoch_loss=%s" % (epoch, train_ppl, train_epoch_loss, eval_ppl, eval_epoch_loss))

    from huggingface_hub import login
    login(token=base64.b64decode(hf_token).decode())

    peft_model_id = peft_model_publish_id
    model.save_pretrained("output_dir", safe_serialization=False)
    model.push_to_hub(peft_model_id, use_auth_token=True, safe_serialization=False)

@component(
    packages_to_install=["kubernetes"],
    base_image='python:3.10'
)
def deploy_modelmesh_custom_runtime(huggingface_name: str, peft_model_publish_id: str, model_name_or_path: str, server_name: str, namespace: str, image: str):
    import kubernetes.config as k8s_config
    import kubernetes.client as k8s_client
    from kubernetes.client.exceptions import ApiException

    def create_custom_object(group, version, namespace, plural, manifest):
        cfg = k8s_client.Configuration()
        cfg.verify_ssl=False
        cfg.host = "https://kubernetes.default.svc"
        cfg.api_key_prefix['authorization'] = 'Bearer'
        cfg.ssl_ca_cert = '/var/run/secrets/kubernetes.io/serviceaccount/ca.crt'
        with open("/var/run/secrets/kubernetes.io/serviceaccount/token") as f:
            lines = f.readlines()
            for l in lines:
                cfg.api_key['authorization'] = "{}".format(l)
                break
        with k8s_client.ApiClient(cfg) as api_client:
            capi = k8s_client.CustomObjectsApi(api_client)
            try:
                res = capi.create_namespaced_custom_object(group=group,
                                                           version=version, namespace=namespace,
                                                           plural=plural, body=manifest)
            except ApiException as e:
                # object already exists
                if e.status != 409:
                    raise
    custom_runtime_manifest = {
        "apiVersion": "serving.kserve.io/v1alpha1",
        "kind": "ServingRuntime",
        "metadata": {
            "name": "{}-server".format(server_name),
            "namespace": namespace
        },
        "spec": {
            "supportedModelFormats": [
            {
                "name": "peft-model",
                "version": "1",
                "autoSelect": True
            }
            ],
            "multiModel": True,
            "grpcDataEndpoint": "port:8001",
            "grpcEndpoint": "port:8085",
            "containers": [
            {
                "name": "mlserver",
                "image": image,
                "env": [
                {
                    "name": "MLSERVER_MODELS_DIR",
                    "value": "/models/_mlserver_models/"
                },
                {
                    "name": "MLSERVER_GRPC_PORT",
                    "value": "8001"
                },
                {
                    "name": "MLSERVER_HTTP_PORT",
                    "value": "8002"
                },
                {
                    "name": "MLSERVER_LOAD_MODELS_AT_STARTUP",
                    "value": "true"
                },
                {
                    "name": "MLSERVER_MODEL_NAME",
                    "value": "peft-model"
                },
                {
                    "name": "MLSERVER_HOST",
                    "value": "127.0.0.1"
                },
                {
                    "name": "MLSERVER_GRPC_MAX_MESSAGE_LENGTH",
                    "value": "-1"
                },
                {
                    "name": "PRETRAINED_MODEL_PATH",
                    "value": model_name_or_path
                },
                {
                    "name": "PEFT_MODEL_ID",
                    "value": "{}/{}".format(huggingface_name, peft_model_publish_id),
                }
                ],
                "resources": {
                "requests": {
                    "cpu": "500m",
                    "memory": "4Gi"
                },
                "limits": {
                    "cpu": "5",
                    "memory": "5Gi"
                }
                }
            }
            ],
            "builtInAdapter": {
            "serverType": "mlserver",
            "runtimeManagementPort": 8001,
            "memBufferBytes": 134217728,
            "modelLoadingTimeoutMillis": 90000
            }
        }
    }
    create_custom_object(group="serving.kserve.io", version="v1alpha1",
                         namespace=namespace, plural="servingruntimes",
                         manifest=custom_runtime_manifest)

@component(
    base_image='python:3.10'
)
def inference_svc(model_name: str, namespace: str) -> str :

    inference_service = '''
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: {}
  namespace: {}
  annotations:
    serving.kserve.io/deploymentMode: ModelMesh
spec:
  predictor:
    model:
      modelFormat:
        name: peft-model
      runtime: {}-server
      storage:
        key: localMinIO
        path: sklearn/mnist-svm.joblib
'''.format(model_name, namespace, model_name)

    return inference_service

@component(
    packages_to_install=["transformers", "peft", "torch", "requests"],
    base_image='python:3.10'
)
def test_modelmesh_model(service: str,  namespace: str, model_name: str, input_tweet: str):
    import requests
    import base64
    import json

    url = "http://%s.%s:8008/v2/models/%s/infer" % (service, namespace, model_name)
    input_json = {
        "inputs": [
            {
            "name": "content",
            "shape": [1],
            "datatype": "BYTES",
            "data": [input_tweet]
            }
        ]
    }

    x = requests.post(url, json = input_json)

    print(x.text)
    respond_dict = json.loads(x.text)
    inference_result = respond_dict["outputs"][0]["data"][0]
    base64_bytes = inference_result.encode("ascii")
  
    string_bytes = base64.b64decode(base64_bytes)
    inference_result = string_bytes.decode("ascii")
    print("inference_result: %s " % inference_result)

@component(
    packages_to_install=["kubernetes"],
    base_image='python:3.10'
)
def get_hf_token() -> str:
    from kubernetes import client, config

    config.load_incluster_config()
    core_api = client.CoreV1Api()
    secret = core_api.read_namespaced_secret(name="huggingface-secret", namespace="kubeflow-user-example-com")
    return secret.data["token"]

# Define your pipeline function
@dsl.pipeline(
    name="Serving LLM with Prompt tuning",
    description="A Pipeline for Serving Prompt Tuning LLMs on Modelmesh"
)
def prompt_tuning_pipeline(
    huggingface_name: str = "difince",
    peft_model_publish_id: str = "bloomz-560m_PROMPT_TUNING_CAUSAL_LM",
    model_name_or_path: str = "bigscience/bloomz-560m",
    model_name: str = "vml-demo",
    input_tweet: str = "@nationalgridus I have no water and the bill is current and paid. Can you do something about this?",
    test_served_llm_model: str ="true",
    num_epochs: int = 1
):
    hf_token_task = get_hf_token()
    prompt_tuning_llm = prompt_tuning_bloom( peft_model_publish_id=peft_model_publish_id, 
                                             model_name_or_path=model_name_or_path,
                                             num_epochs=num_epochs,
                                             hf_token=hf_token_task.output)

    deploy_modelmesh_custom_runtime_task = deploy_modelmesh_custom_runtime(
                                                                           huggingface_name=huggingface_name,
                                                                           peft_model_publish_id=peft_model_publish_id,
                                                                           model_name_or_path=model_name_or_path,
                                                                           server_name=model_name, namespace=modelmesh_namespace,
                                                                           image=peft_model_server_image)
    deploy_modelmesh_custom_runtime_task.after(prompt_tuning_llm)

    inference_svc_task = inference_svc(model_name=model_name, namespace=modelmesh_namespace)
    inference_svc_task.after(deploy_modelmesh_custom_runtime_task)
    inference_svc_task.set_caching_options(False)
    
    kserve_launcher_op = comp.load_component_from_url(kserv_component)
    serve_llm_with_peft_task = kserve_launcher_op(action="apply", 
                                                  inferenceservice_yaml=inference_svc_task.output)
    serve_llm_with_peft_task.after(inference_svc_task)
    serve_llm_with_peft_task.set_caching_options(False)

    with dsl.If(test_served_llm_model == 'true'):
        test_modelmesh_model_task = test_modelmesh_model(service=modelmesh_servicename, namespace=modelmesh_namespace,
                                                         model_name=model_name, input_tweet=input_tweet).after(serve_llm_with_peft_task)
        test_modelmesh_model_task.set_caching_options(False)

# Compile the pipeline
Compiler().compile(prompt_tuning_pipeline, pipeline_out_file)
