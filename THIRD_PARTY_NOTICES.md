# Third-party notices

## BiRefNet Lite 512

This app loads the browser-ready ONNX model [`studioludens/birefnet-lite-512`](https://huggingface.co/studioludens/birefnet-lite-512) from Hugging Face at run time. The model repository identifies its license as MIT and attributes the upstream weights and converted export to:

```text
Copyright (c) 2024 ZhengPeng
Copyright (c) 2026 Studio Ludens V.O.F.
```

The model files are not copied into this project. Consult the [model card](https://huggingface.co/studioludens/birefnet-lite-512) and its [license file](https://huggingface.co/studioludens/birefnet-lite-512/tree/main/LICENSE) for the complete license and model details.

The selected FP16 ONNX graph was checked directly: it accepts `input_image` with shape `[1, 3, 512, 512]` and emits the single-channel mask as `output_image` with shape `[1, 1, 512, 512]`. The app also accepts the `logits` alias used in the model card's Quickstart.

## Transformers.js

The app loads Transformers.js version 3.7.2 from [jsDelivr](https://www.jsdelivr.com/package/npm/@huggingface/transformers) at run time. The pinned package identifies its license as Apache-2.0; see the [package license](https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2/LICENSE) and the [Transformers.js repository](https://github.com/huggingface/transformers.js) for source.
