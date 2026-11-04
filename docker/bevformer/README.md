# BEVFormer 统一镜像构建

以 training-suite v2.1.7-rc4 的固定 digest 为基础，使用 Dockerfile 中固定的七个源码 commit，增量安装 2026-10-23 临时镜像所需依赖。仅使用一个 FROM，直接保留 Base 的系统、PyTorch / torch_musa 2.7.1.post1、NumPy 1.26.4 和 MUSA SDK，不重新构建这些组件。只编译 MMCV、MMDetection3D 和自定义扩展，目标架构为 MUSA 3.1。

在本目录执行（需要支持 BuildKit 的 Docker）：

```bash
# BASE_REPOSITORY 为可访问的 Base 仓库地址，不含 tag。
# GITHUB_TOKEN 需要能读取这七个仓库，包括两个私有扩展仓库。
DOCKER_BUILDKIT=1 docker build \
  --build-arg BASE_REPOSITORY="$BASE_REPOSITORY" \
  --build-arg MAX_JOBS=4 \
  --secret id=github_token,env=GITHUB_TOKEN \
  -t training-suite:2.1.7-rc4-bevformer .
```

凭据只通过 BuildKit secret 使用，不写入源码或镜像层。编译目录为 `/opt/src`，安装完成后清理；BEVFormer 工作目录为 `/opt/bevformer`。DCNv2 与 MSDA 分开编译，避免自动转译覆盖 MSDA 源码；使用通用 x86-64 编译参数。依赖安装使用固定版本、`--no-deps` 和 `--no-build-isolation`，避免自动升级 Base 中的框架包。构建依赖由 Base 和依赖清单提供，关闭旧式 `setup_requires` 的隐式下载。

构建不需要挂载训练数据或 GPU。数据路径在运行时配置，训练配置中的 `workers_per_gpu=4` 保持不变。仅重建所列环境，不包含 10 月 23 日之后的功能修改，也不承诺生成与原镜像相同的二进制文件或镜像 digest。
