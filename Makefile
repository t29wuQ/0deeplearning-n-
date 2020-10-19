run:
	docker run -itd --name deeplearning -v $PWD:/src -w /src python:3.7
rm:
	docker stop deeplearning && docker rm deeplearning
