- kmeans dgsf:
```bash
docker build -t kmeans-dgsf-aws --build-arg LOCAL_FUNC_DIR=./kmeans-dgsf -f ./template/Dockerfile .
```
- face detection
```bash
docker build -t face-detection-aws --build-arg LOCAL_FUNC_DIR=./face-detection -f ./template/Dockerfile .

- covid
```bash
docker build -t covid-aws --build-arg LOCAL_FUNC_DIR=./covid -f ./template/Dockerfile .
```
```




# Build Kmeans

docker build -t kmeans-dgsf-aws --build-arg LOCAL_FUNC_DIR=./kmeans-dgsf -f ./template/Dockerfile .
docker tag kmeans-dgsf-aws:latest YOUR PASSWORD HERE.dkr.ecr.us-east-1.amazonaws.com/dgsf:kmeans
docker push YOUR PASSWORD HERE.dkr.ecr.us-east-1.amazonaws.com/dgsf:kmeans


## Test:

docker run -p 9000:8080 kmeans-dgsf-aws
curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" -d '{}'