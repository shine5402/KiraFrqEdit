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

SwiftF0 ships with its model (MIT) bundled, so it works with no download. An on-disk
`swiftf0.onnx` next to the executable (or in `KIRAFRQ_ML_DIR`) overrides the bundled copy; the
license for the bundled model is next to it at
[`crates/kirafrq-ml-provider/assets/swiftf0.LICENSE`](crates/kirafrq-ml-provider/assets/swiftf0.LICENSE).

## GPU runner (Windows)

The ML estimators run on a DirectX 12 GPU through DirectML when the machine has one; the
CPU is the fallback. The GUI's Advanced settings (hamburger menu) pick the runner and
adapter, and remember the choice across restarts. On the CLI:

```
kirafrqgen-cli --list-ml-devices                        # adapters and their indices
kirafrqgen-cli bank                                     # DirectML when available, else CPU
kirafrqgen-cli bank --ml-runner cpu                     # force the processor
kirafrqgen-cli bank --ml-runner directml --ml-device 1  # pin a specific adapter
```

ONNX Runtime keeps its CPU execution provider alongside DirectML, so operations DirectML
does not take run on the CPU. DIO and Harvest always run on the CPU. The Windows release
binaries bundle Microsoft's DirectML runtime (see [CREDITS.md](CREDITS.md)).

## License

[MIT](LICENSE)
