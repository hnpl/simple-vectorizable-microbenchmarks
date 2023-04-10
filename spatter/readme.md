## Usage

```sh
# Generating binary for the native hardware
make -f Makefile-hw

# Generating binary for RISC-V 64-bit ISA without vector instructions, and with gem5's ROI annotation
M5OPS_HEADER_PATH=../gem5/include/ M5_BUILD_PATH=../gem5/util/m5/build/riscv/ make -f Makefile-rv64gc

# Generating binary for RISC-V 64-bit ISA with vector instructions, and with gem5's ROI annotation
M5OPS_HEADER_PATH=../gem5/include/ M5_BUILD_PATH=../gem5/util/m5/build/riscv/ make -f Makefile-rvv

# Generating binary for ARMv8.1 ISA with SVE instructions, and with gem5's ROI annotation
M5OPS_HEADER_PATH=../gem5/include/ M5_BUILD_PATH=../gem5/util/m5/build/arm/ make -f Makefile-sve
```
