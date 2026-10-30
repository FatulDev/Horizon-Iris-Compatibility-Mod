#!/bin/bash
# run_test.sh OUTDIR [ENV=VAL...]: one GPU test run of the staged build.
OUT=$1; shift; mkdir -p $OUT
cd $HOME/hz-stage && env "$@" VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json IRIS_TEST_PACK=${PACK:-$HOME/Eclipse-Shader} IRIS_TEST_PNG=$OUT CARGO_TARGET_DIR=$HOME/hz-stage-target timeout 3000 cargo test --offline -q -p horizon-client-render --release --lib iris_tests::tests::runs_a_real_pack -- --ignored --nocapture 2>&1 | grep -E "picture|^error|GPU error" -A3 | grep -v "^\.\|test result"
