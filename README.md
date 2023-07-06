# General Congestion Attack on HTLC-Based Payment Channel Networks

This repository contains code for simulating the General Congestion Attack on HTLC-Based Payment Channel Networks as described in the paper ["General Congestion Attack on HTLC-Based Payment Channel Networks"](https://eprint.iacr.org/2020/456).

## Base Repository

This code is built upon the [LNTrafficSimulator](https://github.com/ferencberes/LNTrafficSimulator) repository by Ferenc Béres. Please follow the instructions in the [LNTrafficSimulator's README](https://github.com/ferencberes/LNTrafficSimulator/blob/master/README.md) to download the necessary Lightning Network data.

## How to Run

Execute the following command to run the simulation:

```shell
python Dos_simulation.py preprocessed 0 params.json ./result
