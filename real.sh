#!/bin/bash
for d in sachs syntren
do  
    # joint
    python -m daguerreo.run_model --joint --pruning_reg=0.01 --lr_theta=0.1 --dataset=$d --structure=sp_map --equations=linear --standardize
    python -m daguerreo.run_model --joint --pruning_reg=0.001 --lr_theta=0.1 --dataset=$d --structure=sp_map --equations=nonlinear --standardize
    python -m daguerreo.run_model --joint --pruning_reg=0.001 --lr_theta=0.1 --dataset=$d --structure=tk_sp_max --equations=linear --standardize
    python -m daguerreo.run_model --joint --pruning_reg=0.0001 --lr_theta=0.1 --dataset=$d --structure=tk_sp_max --equations=nonlinear --standardize

    # bi-level
    for e in linear nonlinear
    do
        for s in "sp_map" "tk_sp_max"
        do
            python -m daguerreo.run_model --pruning_reg=0.01 --lr_theta=0.1 --dataset=$d --structure=$s --equations=$e --standardize
        done
    done

    # LARS
    python -m daguerreo.run_model --lr_theta=0.1 --dataset=$d --structure=sp_map --equations=lars --sparsifier=none --standardize --nogpu
    python -m daguerreo.run_model --lr_theta=0.05 --dataset=$d --structure=tk_sp_max --equations=lars --sparsifier=none --standardize --nogpu
done