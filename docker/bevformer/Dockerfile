ARG BASE_REPOSITORY=training-suite
FROM ${BASE_REPOSITORY}:2.1.7-rc4@sha256:9c7f3caa16a29c83381d29f0f5f6e3d7562a86d4f8c15a23e486e2dc672d4b38 AS source

COPY fetch_source.sh /tmp/fetch_source.sh
RUN --mount=type=secret,id=github_token,required=true \
    bash /tmp/fetch_source.sh mmcv 7a9e1c36a4a45f47ff86fc52c0f1c152789e1217 && \
    bash /tmp/fetch_source.sh BEVFormer f173220fb2b8504c93d54ebb6920446fa5ae26d9 && \
    bash /tmp/fetch_source.sh mmdetection a3a31083aae74db0303acc7d45b07108471e89db && \
    bash /tmp/fetch_source.sh mmsegmentation 622c93028b2583588612714374e033411bd20257 && \
    bash /tmp/fetch_source.sh mmdetection3d 664a867a26278356d3103f9e0908b4bc4e9467e3 && \
    bash /tmp/fetch_source.sh bev-kernels ba568b138c81bd5434d0c136813e9ddae5390b45 && \
    bash /tmp/fetch_source.sh bevformer-musa-fusion ec1c1577b5fb4764d1543809b2bd1255d2f9b115

FROM ${BASE_REPOSITORY}:2.1.7-rc4@sha256:9c7f3caa16a29c83381d29f0f5f6e3d7562a86d4f8c15a23e486e2dc672d4b38 AS build
ARG MAX_JOBS=4
ENV TORCH_MUSA_ARCH=31 TORCH_MUSA_ARCH_LIST=31 MATE_MUSA_ARCH_LIST=3.1 \
    FORCE_MUSA=1 MMCV_WITH_OPS=1 MSDA_CROSS_BUILD=1 \
    PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY requirements.txt build_extension.py /tmp/build/
COPY --from=source /opt/src /opt/src

# Keep the framework and SDK supplied by Base; install only the fixed additions.
RUN python -m pip download --only-binary=:all: --no-deps \
        -r /tmp/build/requirements.txt -d /opt/wheels && \
    python -m pip install --no-index --no-deps /opt/wheels/*.whl

# torchada regenerates csrc_musa for DCNv2, so preserve MSDA separately first.
RUN mkdir -p /opt/msda/csrc_musa /opt/build-stubs && \
    cp -a /opt/src/bev-kernels/msda_musa_opt2 /opt/msda/ && \
    cp -a /opt/src/bev-kernels/csrc_musa/msda /opt/msda/csrc_musa/ && \
    ln -s /usr/local/musa/lib/stubs/libmusa.so /opt/build-stubs/libmusa.so.1 && \
    export LD_LIBRARY_PATH=/opt/build-stubs:${LD_LIBRARY_PATH:-} MAX_JOBS=${MAX_JOBS} PIP_NO_INDEX=1 && \
    for source in \
        /opt/src/bevformer-musa-fusion \
        /opt/src/mmcv \
        /opt/src/mmdetection \
        /opt/src/mmsegmentation \
        /opt/src/mmdetection3d \
        /opt/src/bev-kernels \
        /opt/msda/msda_musa_opt2; do \
        cd "$source" && \
        python /tmp/build/build_extension.py setup.py build_ext --force bdist_wheel --dist-dir /tmp/wheel && \
        python -m pip install --no-index --no-deps /tmp/wheel/*.whl && \
        mv /tmp/wheel/*.whl /opt/wheels/ || exit 1; \
    done

FROM ${BASE_REPOSITORY}:2.1.7-rc4@sha256:9c7f3caa16a29c83381d29f0f5f6e3d7562a86d4f8c15a23e486e2dc672d4b38
COPY --from=build /opt/wheels /tmp/wheels
RUN python -m pip install --no-index --no-deps --no-cache-dir /tmp/wheels/*.whl && \
    rm -rf /tmp/wheels
COPY --from=source /opt/src/BEVFormer /opt/bevformer
ENV PYTHONPATH=/opt/bevformer BEVFORMER_SELECTED_CONFIG=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /opt/bevformer
