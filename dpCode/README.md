# Secure Noise Sampling in MPC for DP

## Setup

### Install dependencies

Install [EMP](https://github.com/emp-toolkit/emp-tool), [OpenFHE](https://github.com/openfheorg/openfhe-development) and [Scalable Mixed-Mode MPC](https://github.com/radhika1601/ScalableMixedModeMPC.git)

### Build

```
cd dpCode;
mkdir build; cd build;
cmake ..
make -j4
```

## Benchmark

To run the tests use the following command:

```
./build/bin/test_bench <party> <port> <num-party> <ip-config path> <log-table-size> <statistical-sec>
```

### IP Config

To run the tests across multiple servers you can pass the ip configuration as a txt file with the following format.

```
<ip1> <port1>
<ip2> <port2>
```

### Prior Work

For comparison use code from [Secure Sampling Benchmark](https://github.com/yuchengxj/Secure-sampling-benchmark) with [MP-SPDZ](https://github.com/data61/MP-SPDZ)
