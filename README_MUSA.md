# BEVFormer MUSA 客户运行说明

本仓库提供已交付 MUSA 镜像所对应的 BEVFormer 优化源码。客户在该镜像中挂载本仓库即可使用；运行依赖、修改版 OpenMMLab 和 custom 算子继续使用镜像内的版本。已核实镜像基线为 training-suite 2.1.7-rc4，torch 和 torch_musa 分发包版本为 2.7.1.post1；源码构建的模块版本可能附带后缀。源码来源及文件校验记录见 [source-provenance.json](docs/musa/source-provenance.json)。

**本次交付是源码迁移与静态校验，未针对本 checkout 新做 GPU 训练、精度或性能验证。** 环境检查通过也只表示导入和算子注册正常。

## 1. 使用已交付镜像

沿用已交付镜像的 MUSA 设备、驱动和运行时配置，在容器启动参数中增加仓库、数据、权重和输出目录的挂载。例如使用以下容器内路径（左侧宿主路径由客户替换）：

```bash
-v /host/BEVFormer:/workspace/BEVFormer \
-v /host/nuscenes:/workspace/BEVFormer/data/nuscenes:ro \
-v /host/checkpoints:/workspace/BEVFormer/ckpts:ro \
-v /host/bevformer-output:/workspace/BEVFormer/work_dirs
```

这是一组挂载参数，需加到现有容器启动命令中。双机应让容器间能够访问训练主节点地址和端口；使用主机网络时沿用平台允许的网络配置。无需重建镜像、重新安装依赖或编译算子。

**不要执行上游安装教程中的 `pip install`、`pip install -r requirements.txt`，也不要用官方 wheel 覆盖镜像内的 torch、torch_musa、mmcv、mmdet、mmdet3d、mmseg 或 custom 算子。** 相同版本号不代表具有相同的 MUSA 修改。上游 README 和安装文档仍保留，MUSA 客户以本文为入口。

## 2. 配置、数据和权重

默认使用 [bevformer_base.py](projects/configs/bevformer/bevformer_base.py)，每卡 batch 为 3，即 `data.samples_per_gpu=3`。DCN 重算默认开启（`bev_opt.dcn_recompute_cols=True`），以反向重算换取显存节省；DCN 层保持开启，具体配置是 `stage_with_dcn=(False, False, True, True)`，`model.img_backbone.dcn.backend='fuse_c'`。默认优化开关在配置末尾的 `bev_opt` 中，启动后由 `tools/bev_opt_setup.py` 映射到各进程环境变量。

默认路径均相对于本仓库根目录：

- 数据根目录：`./data/nuscenes/`，包含对应 nuScenes 数据。
- 训练标注：`./data/nuscenes/nuscenes_infos_temporal_train.pkl`。
- 验证/测试标注：`./data/nuscenes/nuscenes_infos_temporal_val.pkl`。
- 预训练权重：`./ckpts/r101_dcn_fcos3d_pretrain.pth`，由配置的 `load_from` 指定。

客户负责准备数据和权重。标注中的文件路径必须能在容器内解析；对照已有训练时应使用相同数据、相同标注及相同预训练权重。双机必须使用相同仓库版本和配置，并在两台容器中提供相同的数据、标注和权重路径。启动脚本会切换到仓库根目录，因此从其他工作目录调用时相对路径含义也保持一致。

可直接调整配置中的路径；也可在训练命令末尾追加 `--cfg-options load_from=/weights/model.pth`。覆盖数据位置时必须同时覆盖 `data.train.data_root`、`data.val.data_root`、`data.test.data_root` 及各自的 `ann_file`；仅覆盖顶层 `data_root` 不会重新计算已经生成的标注路径。

## 3. 只读环境检查

在已交付镜像内执行：

```bash
python -B /workspace/BEVFormer/tools/check_musa_env.py
```

检查会打印实际 Python、框架模块版本、分发包版本和模块路径（保留源码构建的版本后缀），确认 `projects` 和插件路径来自当前 checkout，并导入 `mmdet3d.ops`、`_modulated_deform_conv_ext`、`_msda_opt2_musa`，检查 MMCV MSDA 入口与 fusion 的 PrivateUse1 内核注册。它不导入训练入口、不执行插件初始化、不创建模型、不申请 GPU、不运行算子。

fusion 使用镜像内的 `bevformer_musa_fusion` loader：repo 的融合卷积模块导入它后，loader 找到包内预编译的 `_native*.so`，通过 `torch.ops.load_library` 注册 `musa_fused::conv2d_bias_act_add`，并检查 PrivateUse1 实现。loader 还校验与已交付 torch/torch_musa 构建的兼容性。这里加载已有库，不进行现场编译；检查失败应核对镜像和模块来源。

镜像可能默认设置 `PYTHONPATH=/opt/bevformer`。本仓库启动脚本会把当前 checkout 的绝对路径放到 `PYTHONPATH` 首位，保留其余依赖路径。环境检查也优先使用自身所在的 checkout。

## 4. 单机 8 卡

在容器内执行；配置和 `--work-dir` 的相对路径均相对于仓库根目录：

```bash
cd /workspace/BEVFormer
mkdir -p work_dirs/customer-single
set -o pipefail
bash tools/dist_train.sh projects/configs/bevformer/bevformer_base.py 8 \
  --work-dir work_dirs/customer-single \
  2>&1 | tee work_dirs/customer-single/launcher.log
```

也可从任意目录执行 `bash /workspace/BEVFormer/tools/dist_train.sh ...`。默认 `NNODES=1`、`NODE_RANK=0`、`MASTER_ADDR=127.0.0.1`、`MASTER_PORT=28509`。同机有多个任务时指定不同端口；兼容旧 `PORT` 变量，显式 `MASTER_PORT` 优先。

## 5. 双机，每机 8 卡

两台机器都进入已交付镜像，同时执行各自命令。以下 `192.0.2.10` 仅为示例，必须替换为两台容器均可访问的 rank-0 主机地址。两端使用相同 `MASTER_ADDR`、`MASTER_PORT` 和 `NNODES`，`NODE_RANK` 分别为 0、1。跨节点通信与设备访问沿用已交付平台配置。

节点 0：

```bash
cd /workspace/BEVFormer
mkdir -p work_dirs/customer-dual
set -o pipefail
NNODES=2 NODE_RANK=0 MASTER_ADDR=192.0.2.10 MASTER_PORT=28509 \
  bash tools/dist_train.sh projects/configs/bevformer/bevformer_base.py 8 \
  --work-dir work_dirs/customer-dual \
  2>&1 | tee work_dirs/customer-dual/launcher-node0.log
```

节点 1：

```bash
cd /workspace/BEVFormer
mkdir -p work_dirs/customer-dual
set -o pipefail
NNODES=2 NODE_RANK=1 MASTER_ADDR=192.0.2.10 MASTER_PORT=28509 \
  bash tools/dist_train.sh projects/configs/bevformer/bevformer_base.py 8 \
  --work-dir work_dirs/customer-dual \
  2>&1 | tee work_dirs/customer-dual/launcher-node1.log
```

batch 3 指每卡 batch，单机全局 batch 为 24，双机为 48；启动脚本不会自动调整学习率。上述双机命令保留配置中的学习率，训练策略如需调整由客户显式配置。

## 6. 输出和默认环境

训练日志、配置副本和 checkpoint 写到 `--work-dir`，不指定时写到 `./work_dirs/<配置文件名>/`。目录应挂载到宿主机持久存储；两端使用相同输出路径，分节点的 launcher 日志使用不同文件名。重跑时使用新的输出目录，恢复训练可追加 `--resume-from /path/to/checkpoint.pth`。

启动脚本默认设置 `BEVFORMER_SELECTED_CONFIG=1`，让本次 BEVFormer 配置只加载所需模块，避开未使用的 DD3D 导入；确有其他配置需求时可显式覆盖。保留 `MUSA_EXECUTION_TIMEOUT=3600000` 毫秒默认值，以覆盖 rank 0 验证期间其他 rank 的较长等待；也可显式设置。环境检查、启动参数模拟和语法检查均不能代替客户环境中的实际训练验收。
