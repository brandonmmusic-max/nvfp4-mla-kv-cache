# Builds the nvfp4_ds_mla-capable serving image FROM the public GLM-5.2 hybrid base.
# Base image (weights loader + b12x kernels) is madeby561's work; this adds the
# NVFP4 MLA KV writer (.so) + the ported b12x MLA nvfp4 readers.
FROM madeby561/vllm-glm52-nvfp4-nf3-hybrid:v2 AS build
WORKDIR /tmp/ext
COPY kernel/nvfp4_mla_ext.cu kernel/setup_nvfp4_ext.py ./
RUN python setup_nvfp4_ext.py build_ext --inplace \
    && mkdir -p /opt/nvfp4_extra && cp _C_nvfp4.so /opt/nvfp4_extra/_C_nvfp4.so

FROM madeby561/vllm-glm52-nvfp4-nf3-hybrid:v2
# 1) writer op (.so) + custom-ops loader patch
COPY --from=build /opt/nvfp4_extra/_C_nvfp4.so /opt/nvfp4_extra/_C_nvfp4.so
COPY kernel/patch_custom_ops.py /tmp/patch_custom_ops.py
RUN python /tmp/patch_custom_ops.py && rm /tmp/patch_custom_ops.py
# 2) ported b12x MLA nvfp4 readers + vllm dtype plumbing (overlay onto site-packages)
COPY reader/ /opt/venv/lib/python3.12/site-packages/
