# KiraFrqEdit

Work in progress. Nothing here is ready to use yet, and anything — interfaces, file formats,
behavior — can change without notice.

## Credits

Pitch analysis is [WORLD](https://github.com/mmorise/World) (DIO / Harvest, StoneMask
refinement) by M. Morise and contributors, vendored under `third_party/World`.

The ML estimator runs [RMVPE](https://github.com/Dream-High/RMVPE) by H. Wei, X. Cao, T. Dan
and Y. Chen ([Interspeech 2023](https://arxiv.org/abs/2306.15412)) through
[ONNX Runtime](https://onnxruntime.ai/) (MIT), linked via the
[`ort`](https://github.com/pykeio/ort) crate.

[CREDITS.md](CREDITS.md) is the single credits source, including every Rust dependency and its
license text. `kirafrqgen-cli --license` prints it (add `--license-full` for the license texts),
and the GUI shows it under Help > Credits. Regenerate it with `cargo xtask credits` after a
dependency change.

## ML model

The RMVPE weights are not redistributed with this project. You can get a usable `rmvpe.onnx`
(361 MB) here:
[lj1995/VoiceConversionWebUI/rmvpe.onnx](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.onnx).
Put the file next to the executable (or in the directory named by `KIRAFRQ_ML_DIR`).

## License

[MIT](LICENSE)
