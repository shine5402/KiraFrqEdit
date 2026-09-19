# KiraFrqEdit

Work in progress. Nothing here is ready to use yet, and anything — interfaces, file formats,
behavior — can change without notice.

## Credits

Pitch analysis is [WORLD](https://github.com/mmorise/World) (DIO / Harvest, StoneMask
refinement) by M. Morise and contributors, vendored under `third_party/World`.

The ML estimator runs [RMVPE](https://github.com/Dream-High/RMVPE) by H. Wei, X. Cao, T. Dan
and Y. Chen ([Interspeech 2023](https://arxiv.org/abs/2306.15412)) through
[ONNX Runtime](https://onnxruntime.ai/) (MIT), linked via the
[`ort`](https://github.com/pykeio/ort) crate. Other dependencies are the Rust crates listed in
the `Cargo.toml` files.

## ML model

The RMVPE weights are not redistributed with this project. Download `rmvpe.onnx` from the
mirror the ecosystem uses, on Hugging Face:
[lj1995/VoiceConversionWebUI/rmvpe.onnx](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.onnx).
Put the file next to the executable (or in the directory named by `KIRAFRQ_ML_DIR`). The
[ML estimator notes](docs/research/ml-f0-estimators.md) record the model's provenance and
licensing.

## License

[MIT](LICENSE)
