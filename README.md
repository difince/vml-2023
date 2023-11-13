# Train prompt tuning configuration against LLM and Serve them using Kubeflow Pipelines

This repository demonstrates how Kubeflow could be leveraged for prompt tuning a foundational large language model (LLM) and serving the prompt-tunned model. 
Specifically:
1. Train ( with the use of [PEFT](https://huggingface.co/docs/peft/index) and  [RAFT](https://huggingface.co/datasets/ought/raft) dataset) a prompt tuning configuration against a LLM - [bigscience/bloomz-560m](https://huggingface.co/bigscience/bloomz-560m#model-summary)
2. Publish the trained configuration to HuggingFace.
3. Serve the prompt tuning configuration along with HuggingFace open source LLM.
4. Automate the above steps with Kubeflow Pipelines


- [Prerequisites](#prerequisites)

- [Kubeflow Installation](#kubeflow-installation)
- [KServe Modelmesh Installation](#kserve-modelmesh-installation)
- [Service Account Permissions](#adjust-service-account-permissions)
- [HuggingFace Token](#create-k8s-secret-with-your-hugging-face-account-token)
- [Access Kubeflow UI](#access-kubeflow-ui)
- [Create PodDefault resource](#create-poddefault-resource)
- [Run prompt tuning pipeline](#run-prompttuning-pipeline)
  - [Upload IR yaml file](#1-use-ir-intermideate-representation-yaml-file)
  - [Import Notebook into Kubeflow JupyterLab](#2-use-the-jupiter-noteboook)

## Prerequisites

To successfully run the example provided in this repository, Kubeflow cluster and KServe ModelMesh need to be brought up. Before installing them, you must have the following dependencies installed in a local environment.
- Python 3.9+
- Docker
- Kubectl
- Kustomize 5.0.0
- Install Kubernetes locally (could be done via `kind`, `minikube`, `docker desktop for Mac`)

## Kubeflow Installation

This example has been tested with Kubeflow version 1.8. 

To install Kubeflow, clone the [Manifests repo](https://github.com/kubeflow/manifests) and run the installation using kustomize
```bash
git clone --branch v1.8-branch https://github.com/kubeflow/manifests.git && cd manifests
while ! kustomize build example | awk '!/well-defined/' | kubectl apply -f -; do echo "Retrying to apply resources"; sleep 10; done
```

Before continuing, make sure to wait for all pods in `kubeflow` namespace to become ready
```bash
kubectl -n kubeflow wait --for=condition=Ready pods --all --timeout=1200s
```

## KServe Modelmesh Installation

ModelMesh Serving is the Controller for managing ModelMesh, a general-purpose model serving management/routing layer. The instructions for installing it are provided below. For more detailed information on how to get started, check out this [link](https://github.com/kserve/modelmesh-serving/blob/main/docs/quickstart.md)

1. Clone Modelmesh serving repository
```bash
RELEASE="release-0.11"
git clone -b $RELEASE --depth 1 --single-branch https://github.com/kserve/modelmesh-serving.git
cd modelmesh-serving
```
2. Create a namespace called `modelmesh-serving` to deploy ModelMesh to.
```bash
kubectl create namespace modelmesh-serving
./scripts/install.sh --namespace-scope-mode --namespace modelmesh-serving --quickstart
```

## Adjust service account permissions
Give our current service account `kubeflow-user-example-com:default-editor` permissions to manipulate `inferenceservices` and `servingruntimes` within `modelmesh-serving` namespace: 

```bash
kubectl create clusterrole servicemesh-editor --verb=get,create,delete,list,watch,patch --resource=inferenceservices,servingruntime
kubectl create rolebinding servicemesh-editor --serviceaccount=kubeflow-user-example-com:default-editor --clusterrole=servicemesh-editor -n modelmesh-serving
```

## Create k8s secret with your Hugging Face account token

Modify and execute the following command to store your Hugging Face account token as a secret in the Kubeflow  cluster. This secret is used by the pipeline to publish the prompt tuning configuration to Hugging Face after the training. You can obtain the Hugging Face account token with WRITE permission on their [website](https://huggingface.co/settings/tokens).
```
kubectl create secret generic huggingface-secret --from-literal='token=<HuggingFace_WRITE_Token>' -n kubeflow
```
## Access Kubeflow UI

After successful installation of Kubeflow, use `port-forward` to expose the `istio-ingressgateway` service by running the following
  ```
  kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80
  ```
Navigate to [localhost:8080](http://localhost:8080/) and login using `user@example.com` and password `12341234`

## Run PromptTuning Pipeline
###  Use IR (Intermediate Representation) Yaml [file](./llm-prompt_tuning_pipeline.yaml)
You could create the prompt tuning pipeline by uploading the ready to use IR yaml file through the Kubeflow Dashboard. Go to "Pipelines" -> "Upload Pipelines" and select "Upload from file". 
Then create a run from the pipeline.
### Use the Jupiter [Notebook](./prompt_tunning_pipeline.ipynb)
You could leverage the notebook to create the prompt tuning pipeline.  To do so, first you need to  
- Create `PodDefault` resource to inject `ServiceAccount` token volume into your Pods. Once created and configured correctly with your notebook, this will allow all pods created by the notebook to access kubeflow pipelines.

    ```
    kubectl apply -f - <<EOF
    apiVersion: kubeflow.org/v1alpha1
    kind: PodDefault
    metadata:
      name: access-kf-pipeline
      namespace: kubeflow-user-example-com
    spec:
      desc: Allow access to KFP
      selector:
        matchLabels:
          access-kf-pipeline: "true"
      volumeMounts:
        - mountPath: /var/run/secrets/kubeflow/pipelines
          name: volume-kf-pipeline-token
          readOnly: true
      volumes:
        - name: volume-kf-pipeline-token
          projected:
            sources:
              - serviceAccountToken:
                  path: token
                  expirationSeconds: 7200
                  audience: pipelines.kubeflow.org
      env:
        - name: KF_PIPELINES_SA_TOKEN_PATH
          value: /var/run/secrets/kubeflow/pipelines/token
    EOF
    ```

- Import [Notebook](./prompt_tunning_pipeline.ipynb) into Kubeflow JupyterLab

    Once logged in to the Kubeflow dashboard, navigate to "Notebooks" to create a new `JupyterLab` notebook with `kubeflownotebookswg/jupyter-tensorflow-full:v1.8.0-rc.0` image and configuration "Allow access to Kubeflow Pipelines" enabled (available in "Advanced options").

    After the notebook is running, `connect` to the notebook and upload the [notebook file](./prompt_tunning_pipeline.ipynb) notebook.
