# FSA: Fusing FlashAttention within a Single Systolic Array
## 仓库源地址
```
https://github.com/VCA-EPFL/FSA
https://github.com:VCA-EPFL/chipyard-fsa
```
## 环境配置
```
# fsa重新构建了一个包含有fsa的chipyrad
git clone git@github.com:VCA-EPFL/chipyard-fsa.git
cd chipyard-fsa
 ./build-setup.sh --skip-ctags --skip-firesim --skip-marshal
```
## 测试
```
# 硬件编译
cd chipyard-fsa/sims/verilator
# Generate a 4x4 Fp16-mul-Fp32-acc systolic array
make CONFIG=FSA4X4Fp16Config
# python + verilator仿真测试
uv run main.py --seq_q 4 --seq_kv 4 --config FSA4X4Fp16Config --diff --diff_verbose
```

```
graph TD
    subgraph Host ["主机 (Host / Python & C++)"]
        Py[Python Compiler] -->|生成指令流| Bin[Binary Instructions]
        Bin -->|TSI 接口| Decoder
    end

    subgraph Hardware ["FSA 硬件 (Chisel Design)"]
        Decoder -->|DMA 指令| DMA[DMA Engine]
        Decoder -->|Matrix 指令| Ctrl[Matrix Controller]

        %% 内存交互
        DRAM["DRAM / Main Memory"] <-->|AXI4 Bus| DMA

        %% 片上存储
        DMA -->|Write Q,K,V| Spad["Scratchpad SRAM"]
        DMA -->|Read O| AccSRAM["Accumulator SRAM"]

        %% 核心计算路径
        Spad -->|Read| InputDelayer
        InputDelayer --> SA["Systolic Array<br/>(PEs + CMP)"]

        Ctrl -.->|Control Signals| Spad
        Ctrl -.->|Control Signals| SA
        Ctrl -.->|Control Signals| AccUnit
        Ctrl -.->|Control Signals| AccSRAM

        %% 回写路径
        SA -->|Partial Sums| AccUnit[Accumulator]
        AccUnit <-->|Read/Write| AccSRAM

        %% 同步机制
        DMA -- Sync (Semaphore) --> Ctrl
    end

    %% 数据流标注
    linkStyle 4 stroke-width:2px,fill:none,stroke:blue
    linkStyle 5 stroke-width:2px,fill:none,stroke:blue
    linkStyle 6 stroke-width:2px,fill:none,stroke:green
    linkStyle 7 stroke-width:4px,fill:none,stroke:red
    linkStyle 10 stroke-width:2px,fill:none,stroke:green
```

**FSA executes _every_ FlashAttention operation within a single systolic array — without requiring vector units!**
Enjoy computing non-matrix-multiplication operations using matrix-multiplication FLOPs.

- Attention operations are overlapped element-wise within the systolic array to minimize execution latency.
- FSA achieves 1e-3 accuracy compared to `torch.nn.functional.scaled_dot_product_attention` on `fp16`.

![Inner loop animation](docs/innerloop.gif)

Please check the [paper](http://arxiv.org/abs/2507.11331) for more details.

---

## 🚀 Setup

> **Note:** Do **not** clone this repository directly. The commands below will automatically clone FSA as a submodule under `chipyard-fsa/generators/`.

FSA depends on [Chipyard](https://github.com/ucb-bar/chipyard),
and Chipyard requires the [Conda](https://docs.conda.io/en/latest/) package manager.
If you don't have Conda installed, please follow the Conda installation documentation
or use the following command:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
```

To use FSA within Chipyard:

```bash
git clone git@github.com:VCA-EPFL/chipyard-fsa.git
cd chipyard-fsa
./build-setup.sh --skip-ctags --skip-firesim --skip-marshal
# Make sure this is executed before running RTL simulation
source env.sh
```

---

## Run RTL Simulation

### 1. Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Generate the Verilator simulator binary:

```bash
cd chipyard-fsa/sims/verilator
# Generate a 4x4 Fp16-mul-Fp32-acc systolic array
make CONFIG=FSA4X4Fp16Config
```

See [`FSAConfig.scala`](https://github.com/VCA-EPFL/chipyard-fsa/blob/msaga-main/generators/chipyard/src/main/scala/config/FSAConfig.scala) for more available configurations.

### 3. Run FlashAttention using the FSA Python API:

The FlashAttention kernel for FSA is in the file [main.py](python/main.py). To run it, simply use the following commands:

```bash
cd chipyard-fsa/generators/fsa/python
uv run main.py --seq_q 4 --seq_kv 4 --config FSA4X4Fp16Config
```

### 4. (Optional) Value-by-value floating-point error checking:

FSA uses hardware floating-point arithmetic from [EasyFloat](https://github.com/VCA-EPFL/easyfloat), which simplifies subnormal handling compared to [HardFloat](https://github.com/ucb-bar/berkeley-hardfloat).

A Python software library also serves as a *golden reference*, allowing value-by-value comparison with hardware results.

To enable detailed error checking, use:

```bash
cd chipyard-fsa/generators/fsa/python
uv run main.py --seq_q 4 --seq_kv 4 --config FSA4X4Fp16Config --diff --diff_verbose
```

Example output:

```
Comparing with Torch...
Error of FSA vs torch:         {'MAE': np.float32(9.6587464e-05), 'MSE': np.float32(1.6099552e-08), 'MaxErr': np.float32(0.00030440092), 'RelErr': np.float32(0.00019886618)}
Error of PyEasyFloat vs torch: {'MAE': np.float32(9.6587464e-05), 'MSE': np.float32(1.6099552e-08), 'MaxErr': np.float32(0.00030440092), 'RelErr': np.float32(0.00019886618)}
```

---

## FSA Architecture
![microarch](./docs/microarch.jpg)

The RTL implementation is located under the [src](./src/) folder.
The top module is [AXI4FSA.scala](./src/main/scala/fsa/AXI4FSA.scala).
The hardware behavior of FSA instructions is described
in [ExecutionPlan.scala](./src/main/scala/fsa/ExecutionPlan.scala),
and the control logic is generated automatically accordingly.

### Integration Options

We provide two options to integrate FSA into Chipyard:

1. **TileLink Integration**: Connect FSA AXI4 memory channels to Chipyard's TileLink MBus.
   This should be used if FSA shares the backing memory with other
   Chipyard components. The corresponding configs are named
   `FSAMxNConfig` in [`FSAConfig.scala`](https://github.com/VCA-EPFL/chipyard-fsa/blob/msaga-main/generators/chipyard/src/main/scala/config/FSAConfig.scala).

2. **Direct AXI4 Integration**: Connect FSA AXI4 memory channels to the backing memory (e.g., DRAMSim, HBM)
   directly without converting AXI4 to TileLink.
   This is recommended if FSA does not need to share the MBus.
   The corresponding configs are named
   `AXI4FSAMxNConfig` in [`FSAConfig.scala`](https://github.com/VCA-EPFL/chipyard-fsa/blob/msaga-main/generators/chipyard/src/main/scala/config/FSAConfig.scala).


## FPGA Support

[AMD U55C FPGA board](https://www.amd.com/en/products/accelerators/alveo/u55c/a-u55c-p00g-pq-g.html) is supported by this project.

![FPGA arch](./docs/fpga.jpg)

### 1. FPGA bit generation

Make sure [Vivado](https://www.amd.com/en/products/software/adaptive-socs-and-fpgas/vivado.html) is installed and run the following commands:

```bash
cd chipyard-fsa/fpga
make SUB_PROJECT=u55c CONFIG=EmptyU55CConfig TOP=EmptyChipTop bitstream
```
Generated bitstream file can be found at `chipyard-fsa/fpga/generated-src/chipyard.fpga.u55c.U55CFPGATestHarness.EmptyU55CConfig/U55CFPGATestHarness.bit`. You can flash this file onto the FPGA board with Vivado. Flashing from a machine other than the one with FPGA card installed (host machine) is highly recommended.

### 2. FPGA Host machine configuration

Make sure that host machine has Xilinx's [XDMA driver](https://github.com/Xilinx/dma_ip_drivers) installed and loaded. The PCIE bus should be rescaned every time we flash a bitstream with the following commands:

```bash
echo 1 > /sys/class/pci_bus/0000:01/device/remove
echo 1 > /sys/bus/pci/rescan
```

You should be able to see `xdma0_c2h_0`, `xdma0_h2c_0`, and `xdma0_user` devices under `/dev` now.

### 3. Run FPGA test

Still on host machine, run the following commands to launch a test:

```bash
cd chipyard-fsa/generators/fsa/python
uv run main.py --seq_q 16 --seq_kv 16 --config EmptyU55CConfig --engine FPGA
```

Example output:

```
Loading config from: ../../../fpga/generated-src/chipyard.fpga.u55c.U55CFPGATestHarness.EmptyU55CConfig/chipyard.fpga.u55c.U55CFPGATestHarness.EmptyU55CConfig.FSAConfig.json
Device finished execution
Performance counters:
Execution time: 9414 cycles
Max bubble cycles: 6535 cycles
Max active cycles: 179 cycles
DMA active cycles: 233 cycles
Raw instructions: 32
Max instructions: 5
DMA instructions: 4
Fence instructions: 1
Enqueue instructions: 32
Dequeue instructions: 32
Reading back output tensor from addr 0x80000600, size 1024
Comparing with Torch...
Error of FSA vs torch: {'MAE': np.float32(9.4124e-05), 'MSE': np.float32(1.3156206e-08), 'MaxErr': np.float32(0.00031119585), 'RelErr': np.float32(0.00018823991)}
```

For another test run, restart from step 2 to reset the FPGA system.
