#!/bin/bash
SCRIPT_NAME=$1
SRC_IMAGES=$2
REPO=/opt/lbm/Lb-Python

mkdir -p ${REPO}/results/freesurface/${SCRIPT_NAME}

# Only copy final snapshots
find "${SRC_IMAGES}" -name "phi_snapshot*"        | sort | tail -1 | xargs -I{} cp {} ${REPO}/results/freesurface/${SCRIPT_NAME}/
find "${SRC_IMAGES}" -name "density_map*"         | sort | tail -1 | xargs -I{} cp {} ${REPO}/results/freesurface/${SCRIPT_NAME}/
find "${SRC_IMAGES}" -name "*channel_parameters*" | xargs -I{} cp {} ${REPO}/results/freesurface/${SCRIPT_NAME}/
find "${SRC_IMAGES}" -name "*Metrics*"            | xargs -I{} cp {} ${REPO}/results/freesurface/${SCRIPT_NAME}/

cd ${REPO}
git add -A
git commit -m "Auto-push ${SCRIPT_NAME} $(date)"
git pull --rebase
git push