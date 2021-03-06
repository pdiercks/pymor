#!/bin/bash

if [ "x${CI_MERGE_REQUEST_ID}" == "x" ] ; then
    export PULL_REQUEST=false
else
    export PULL_REQUEST=${CI_MERGE_REQUEST_ID}
fi

export PYTHONPATH=${CI_PROJECT_DIR}/src:${PYTHONPATH}

PYMOR_ROOT="$(cd "$(dirname ${BASH_SOURCE[0]})" ; cd ../../ ; pwd -P )"
cd "${PYMOR_ROOT}"

# any failure here should fail the whole test
set -eux

docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
docker pull ${IMAGE}
container=$(docker create --entrypoint / ${IMAGE})

PUBLIC_DIR=/tmp/public
mkdir -p ${PUBLIC_DIR}/${CI_COMMIT_REF_SLUG}/
docker cp ${container}:/public/ ${PUBLIC_DIR}/ || echo "No previous docs builds in storage"
du -sch ${PUBLIC_DIR}/*
rm -rf ${PUBLIC_DIR}/${CI_COMMIT_REF_SLUG}/

# we get the already built html documentation as an artefact from an earlier build stage
rsync -a docs/_build/html/ ${PUBLIC_DIR}/${CI_COMMIT_REF_SLUG}/
cp -r docs/public_root/* ${PUBLIC_DIR}
${PYMOR_ROOT}/.ci/gitlab/docs_makeindex.py ${PUBLIC_DIR}
du -sch ${PUBLIC_DIR}/*
docker build -t ${IMAGE} -f .ci/docker/docs/Dockerfile ${PUBLIC_DIR}
docker push ${IMAGE}
# for automatic deploy gitlab uses ${PROJECT_DIR}/public
mv ${PUBLIC_DIR} ${PYMOR_ROOT}/
ls ${PYMOR_ROOT}/public/
