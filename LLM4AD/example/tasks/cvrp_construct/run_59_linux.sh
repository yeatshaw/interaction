#!/usr/bin/env bash
ROOT=/public/home/liuyang/interaction/LLM4AD
export LLM4AD_TASK_SCRIPT="$ROOT/example/tasks/cvrp_construct/run_eoh.py"
export LLM4AD_LOG_ROOT="$ROOT/logs/cvrp_59"
export LLM4AD_CVRP_TEST_DATA=/public/home/liuyang/dataset/cvrplib/data/cvrp_test_lt200.npz
exec bash "$ROOT/example/tasks/run_59_common.sh"
