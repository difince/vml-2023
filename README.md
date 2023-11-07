# Serve large language models (LLMs) with custom prompt tuning configuration using Kubeflow Pipelines

This repository demonstrates how Kubeflow could be leveraged for prompt tuning foundational LLM and serving the tunned models. 
Specifically:
1. Train a prompt tuning configuration against Hugging Face open source model.
2. Publish a trained configuration to HuggingFace.
3. Serve a prompt tuning configuration along with HuggingFace open source large language models (LLM).
4. Automate the above steps with Kubeflow Pipelines


- [Prerequisites](#prerequisites)

- [Kubeflow Installation](#kubeflow-installation)
- [KServe Modelmesh Installation](#kserve-modelmesh-installation)
- [Service Account Permissions](#adjust-service-account-permissions)
- [HuggingFace Token](#create-k8s-secret-with-your-hugging-face-account-token)
- [Create PodDefault resource](#create-poddefault-resource)
- [Access Kubeflow UI](#access-kubeflow-ui)

- [Import Notebook into Kubeflow JupyterLab](#import-notebook-into-kubeflow-jupyterlab)
- [Run through the Notebook](#run-through-the-notebook)

## Prerequisites

To successfully run the example provided in this repository, Kubeflow cluster and KServe ModelMesh need to brought up. Before installing them, you must have the following dependencies installed in local environment.
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

ModelMesh Serving is the Controller for managing ModelMesh, a general-purpose model serving management/routing layer. The instructions for installing it are provided on the bellows. For more detailed information on how to get started, check our this [link](https://github.com/kserve/modelmesh-serving/blob/main/docs/quickstart.md)

1. Clone Modelmesh serving repository
```bash
RELEASE="release-0.11"
git clone -b $RELEASE --depth 1 --single-branch https://github.com/kserve/modelmesh-serving.git
cd modelmesh-serving
```
2. Create a namespace called `modelmesh-serving` to deploy ModelMesh to.
```bash
kubectl create namespace modelmesh-serving
./scripts/install.sh --namespace modelmesh-serving --quickstart
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

## Create PodDefault resource
Create `PodDefault` to inject `ServiceAccount` token volumne into your Pods. Once created and configured correctly with your notebook, this will allow all pods created by the notebook to access kubeflow pipelines.

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

## Access Kubeflow UI

After successful installation of Kubeflow, use `port-forward` to expose the `istio-ingressgateway` service by running the following
  ```
  kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80
  ```

Navigate to [localhost:8080](http://localhost:8080/) and login using `user@example.com` and password `12341234`

## Import Notebook into Kubeflow JupyterLab

Once logged in to the Kubeflow dashboard, navigate to "Notebooks" to create a new `JupyterLab` notebook with `kubeflownotebookswg/jupyter-tensorflow-full:v1.8.0-rc.0` image and configuration "Allow access to Kubeflow Pipelines" enabled (available in "Advanced options").

After notebook is running, `connect` to the notebook and upload [this]() notebook.
