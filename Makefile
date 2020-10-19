run:
	docker run -itd --name deeplearning -v $(shell pwd):/src -w /src python:3.7
	docker exec -it deeplearning pip install numpy
	docker exec -it deeplearning pip install matplotlib
rm:
	docker stop deeplearning && docker rm deeplearning

exec:
	docker exec -it deeplearning python main.py
