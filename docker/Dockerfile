FROM klc/glm52-nvfp4-dcpmtp:v3.3 AS build

WORKDIR /tmp/nvfp4_mla_ext
COPY nvfp4_mla_ext.cu setup_nvfp4_ext.py ./
RUN python setup_nvfp4_ext.py build_ext --inplace \
    && mkdir -p /opt/nvfp4_extra \
    && cp /tmp/nvfp4_mla_ext/_C_nvfp4.so /opt/nvfp4_extra/_C_nvfp4.so

FROM klc/glm52-nvfp4-dcpmtp:v3.3

COPY --from=build /opt/nvfp4_extra/_C_nvfp4.so /opt/nvfp4_extra/_C_nvfp4.so
COPY patch_custom_ops.py /tmp/patch_custom_ops.py
RUN python /tmp/patch_custom_ops.py && rm /tmp/patch_custom_ops.py
