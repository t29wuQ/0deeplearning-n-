run:
	docker run -itd --name deeplearning -v $PWD:/src -w /src python:3.7
	docker exec -it deeplearning pip install numpy
	docker exec -it deeplearning pip install matplotlib
rm:
	docker stop deeplearning && docker rm deeplearning
