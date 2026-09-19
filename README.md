# KiraFrqEdit

Work in progress. Nothing here is ready to use yet, and anything — interfaces, file formats,
behavior — can change without notice.

## Credits

Pitch analysis is [WORLD](https://github.com/mmorise/World) (DIO / Harvest, StoneMask
refinement) by M. Morise and contributors, vendored under `third_party/World`.

The ML estimator runs [RMVPE](https://github.com/Dream-High/RMVPE) by H. Wei, X. Cao, T. Dan
and Y. Chen ([Interspeech 2023](https://arxiv.org/abs/2306.15412)) through
[ONNX Runtime](https://onnxruntime.ai/) (MIT), linked via the
[`ort`](https://github.com/pykeio/ort) crate, and [SwiftF0](https://github.com/lars76/swift-f0)
by Lars Nieradzik ([arXiv:2508.18440](https://arxiv.org/abs/2508.18440), MIT). Other
dependencies are the Rust crates listed in the `Cargo.toml` files.

## ML model

The RMVPE weights are not redistributed with this project. You can get a usable `rmvpe.onnx`
(361 MB) here:
[lj1995/VoiceConversionWebUI/rmvpe.onnx](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.onnx).
Put the file next to the executable (or in the directory named by `KIRAFRQ_ML_DIR`).

SwiftF0 ships with its model (MIT) bundled, so it works with no download. An on-disk
`swiftf0.onnx` next to the executable (or in `KIRAFRQ_ML_DIR`) overrides the bundled copy; the
license for the bundled model is next to it at
[`crates/kirafrq-ml-provider/assets/swiftf0.LICENSE`](crates/kirafrq-ml-provider/assets/swiftf0.LICENSE).

## License

[MIT](LICENSE)
