# Real-time Prometheus

Application that uses your webcam to showcase how to detect early fires with the [Prometheus][prometheus] software

[prometheus]: https://github.com/santiagxf/prometheus

## Prometheus software as a Docker container

$ docker pull santiagof/prometheus:v0.9.0
$ docker run -p 4000:80 santiagof/prometheus:v0.9.0

$ docker run santiagof/prometheus:v0.9.0 ls -l /var/log/

$ docker build -t cntk CNTK-Prometheus
$ docker run -d -p 8888:8888 --name cntk-prometheus-1 -t cntk:latest
$ docker ps
[Grab ps id]
$ docker exec -it [ps id] bash

## References

* http://cv-tricks.com/object-detection/faster-r-cnn-yolo-ssd/
* https://github.com/tzutalin/labelImg
* https://github.com/fchollet/deep-learning-models
* https://github.com/fchollet/deep-learning-with-python-notebooks
* https://github.com/keras-team/keras/tree/master/examples
* https://github.com/tensorflow/models/tree/master/research/object_detection
* https://medium.com/@siddharthdas_32104/cnns-architectures-lenet-alexnet-vgg-googlenet-resnet-and-more-666091488df5

