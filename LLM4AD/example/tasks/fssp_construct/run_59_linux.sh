#!/usr/bin/env bash
ROOT=/public/home/liuyang/interaction/LLM4AD
export LLM4AD_TASK_SCRIPT="$ROOT/example/tasks/fssp_construct/run_eoh.py"
export LLM4AD_LOG_ROOT="$ROOT/logs/fssp_59"
export LLM4AD_FSSP_TEST_DATA=/public/home/liuyang/dataset/Taillard_scheduling/fsp/fsp_instances.npz
exec bash "$ROOT/example/tasks/run_59_common.sh"
