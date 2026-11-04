# BEVFormer 统一镜像构建

以 training-suite 2.1.7-rc4 的固定 digest 为基础，使用 Dockerfile 中固定的七个源码 commit，重建 2026-10-23 临时镜像对应的依赖环境。保留 Base 的 PyTorch / torch_musa 2.7.1.post1、NumPy 1.26.4 和 MUSA SDK；重新编译扩展，目标架构为 MUSA 3.1。

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

凭据只通过 BuildKit secret 使用，不写入源码或镜像层。源码统一放在构建阶段的 `/opt/src`；运行阶段的工作目录为 `/opt/bevformer`。DCNv2 与 MSDA 分开编译，避免自动转译覆盖 MSDA 源码；使用通用 x86-64 编译参数。

构建不需要挂载训练数据或 GPU。数据路径在运行时配置，训练配置中的 `workers_per_gpu=4` 保持不变。仅重建所列环境，不包含 10 月 23 日之后的功能修改，也不承诺生成与原镜像相同的二进制文件或镜像 digest。
