#!/bin/sh

export GITHUB_PERSONAL_TOKEN=${1}
docker build -f Dockerfile -t kogito-runai --build-arg GITHUB_PERSONAL_TOKEN .