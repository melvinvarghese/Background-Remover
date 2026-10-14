# Cutout — background remover, kept local

A small, open-source background remover that runs in your browser. The image you choose is decoded and processed on your device; this app does not upload or store the photo, mask, or result. A transparent PNG is created only when you choose **Download**.

## Try it locally

Open the live app at <https://melvinvarghese.github.io/Background-Remover/>. To run it locally, serve the repository root over HTTP (opening the HTML directly as a `file://` URL will not load browser modules reliably):

```sh
python3 -m http.server 8000
```

Then open <http://localhost:8000> in a current browser. The first removal loads Transformers.js from jsDelivr and downloads the model from Hugging Face (about 94 MB). The browser may cache the model and runtime assets for later runs; selected images are not cached. WebGPU is used when the browser and device support it; otherwise the app uses the WebAssembly runtime.

## Privacy boundary

- The selected image is decoded and processed in the active page. The app sends no image, mask, or result to a server and uses no analytics, cookies, or ad tags.
- The page and AI model are fetched online. The model host and JavaScript CDN can receive ordinary request metadata, but they do not receive the user's photo from this app.
- The app does not write user images to browser storage. A download happens only after the user clicks **Download transparent PNG**.
- Refreshing or closing the page clears the in-memory photo and cutout.

## Quality and limits

- Inputs: JPEG, PNG, or WebP, up to 20 MB and 25 megapixels.
- BiRefNet Lite receives a **512 × 512** model input. The predicted mask is resized to the original image dimensions, so the PNG preserves the uploaded photo's width and height, but the model cannot infer fine detail beyond its 512 × 512 input.
- Edge softness is a light post-processing control, not a manual brush or a guarantee of perfect hair/transparency edges.
- Browser performance depends on the device; CPU fallback can take longer than WebGPU.

## Project license

The application source is licensed under GPL-3.0; see [`LICENSE`](./LICENSE). The selected `studioludens/birefnet-lite-512` model is MIT-licensed and loaded directly from its publisher rather than copied into this repository. See [Third-party notices](./THIRD_PARTY_NOTICES.md).
