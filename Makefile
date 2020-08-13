build-base:
	docker build . -t assistant_intent_classifier/base -f docker_images/base/Dockerfile

run-core:
	docker run assistant_intent_classifier/core
    # Only difference is that one has large files while the other has code

build-core:
	export DOCKER_BUILDKIT=1
	docker build . -t assistant_intent_classifier/core -f docker_images/core/Dockerfile
    # Only difference is that one has large files while the other has code


