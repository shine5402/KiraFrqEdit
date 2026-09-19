# KiraFrqEdit

Work in progress. Nothing here is ready to use yet, and anything — interfaces, file formats,
behavior — can change without notice.

## Credits

Third-party credits, the Rust dependency list and every license text live in
[CREDITS.md](CREDITS.md). `kirafrqgen-cli --license` prints it (add `--license-full` for the
license texts), and the GUI shows it under Help > Credits. Regenerate it with `cargo xtask
credits` after a dependency change.

## ML model

The RMVPE weights are not redistributed with this project. You can get a usable `rmvpe.onnx`
(361 MB) here:
[lj1995/VoiceConversionWebUI/rmvpe.onnx](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.onnx).
Put the file next to the executable (or in the directory named by `KIRAFRQ_ML_DIR`).

## License

[MIT](LICENSE)
