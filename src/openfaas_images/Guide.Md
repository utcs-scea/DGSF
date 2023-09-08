# Deploy K8 and Openfaas

put
{"auths": {"https://index.docker.io/v1/": {"auth": "aGZpbmdsZXI6UnN0OWJrZ2t4dU1SRW05"}}}
in
~/.docker/config.json


sudo apt-get update
sudo apt-get install -y apt-transport-https ca-certificates curl
sudo curl -fsSLo /usr/share/keyrings/kubernetes-archive-keyring.gpg https://packages.cloud.google.com/apt/doc/apt-key.gpg
echo "deb [signed-by=/usr/share/keyrings/kubernetes-archive-keyring.gpg] https://apt.kubernetes.io/ kubernetes-xenial main" | sudo tee /etc/apt/sources.list.d/kubernetes.list
sudo apt-get update
sudo apt-get install -y kubectl
wget -q -O - https://raw.githubusercontent.com/rancher/k3d/main/install.sh | bash

curl -sL https://cli.openfaas.com | sudo sh
curl -SLsf https://dl.get-arkade.dev/ | sudo sh


k3d cluster create mycluster
sleep 5
arkade install openfaas
kubectl rollout status -n openfaas deploy/gateway
kubectl port-forward -n openfaas svc/gateway 8080:8080 &
PASSWORD=$(kubectl get secret -n openfaas basic-auth -o jsonpath="{.data.basic-auth-password}" | base64 --decode; echo)
echo -n $PASSWORD | faas-cli login --username admin --password-stdin
faas-cli store deploy figlet
kubectl apply -f dockerhub_secret.yml
echo "Hello" | faas-cli invoke figlet

# Setup for dgsf

Check all dgsf .yml files for correct AVA_WORKER_DUMP_DIR path.

Update tools/ava.conf to have the correct IP

<!-- # Measure native

sudo emacs /etc/docker/daemon.json
Add:  
 "default-runtime": "nvidia",

kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.10.0/nvidia-device-plugin.yml -->


# Create and test echo

faas new --lang python3-flask echo-json
## edit .yml to have secret and docker hub url
faas up -f echo-json.yml
cat test.json | faas-cli invoke echo-json