create docker image by this Dockerfile(run in project root directory)
nvidia-docker build -t rrunet docker/

run the container from this image with nvidia's cuda support
nvidia-docker run --name rrunet --rm -ti -v /media/ian/WD/:/WD rrunet:latest /bin/bash

if using dataloader with worker number >= 1, add option --shm-size 8G to increase memory share size
