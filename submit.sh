#!/usr/bin/env bash

# Get all gcloud ml-engine arguments
NAME=$1
JOB_DIR=$2
GC_PROJECT=$3
STAGING_BUCKET=$4

# Skip first four arguments; pass the rest as command-line argument to script
shift 4

now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME=${NAME}_${now}

CONFIG=config.yaml
REGION=us-central1

JOB_DIR="${JOB_DIR}/${JOB_NAME}"  # Passed to the script as --job-dir

PACKAGE_PATH=awr
MAIN_MODULE="${PACKAGE_PATH}.train"

echo "Starting job '${JOB_NAME}' on GCE ML Engine..."

gcloud ai-platform jobs submit training ${JOB_NAME} \
    --verbosity debug \
    --project ${GC_PROJECT} \
    --staging-bucket ${STAGING_BUCKET} \
    --package-path ${PACKAGE_PATH} \
    --packages ../pyoneer/dist/fomoro-pyoneer-0.3.0.tar.gz \
    --module-name ${MAIN_MODULE} \
    --config ${CONFIG} \
    --region ${REGION} \
    --job-dir ${JOB_DIR} \
    -- \
    $@
