#!/bin/sh

export GITHUB_PERSONAL_TOKEN=${1}
docker build -f Dockerfile -t ic-registry.epfl.ch/nlp/kogito --build-arg GITHUB_PERSONAL_TOKEN .