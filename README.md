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

## GPU runner

The ML estimators run on the GPU when the machine has a usable one, through DirectML on
Windows (a DirectX 12 GPU) or Core ML on Apple-silicon Macs; the CPU is the fallback. The GUI's
Advanced settings (hamburger menu) pick the runner (and, for DirectML, the adapter) and remember
the choice across restarts. On the CLI:

```
kirafrqgen-cli --list-ml-devices                        # DirectML adapters and their indices
kirafrqgen-cli bank                                     # the GPU runner when available, else CPU
kirafrqgen-cli bank --ml-runner cpu                     # force the processor
kirafrqgen-cli bank --ml-runner directml --ml-device 1  # Windows: pin a specific adapter
kirafrqgen-cli bank --ml-runner coreml                  # Apple silicon: Core ML
```

ONNX Runtime keeps its CPU execution provider alongside the GPU runners, so operations they do
not take run on the CPU. DIO and Harvest always run on the CPU. The Windows release binaries
bundle Microsoft's DirectML runtime (see [CREDITS.md](CREDITS.md)). Core ML caches its compiled
models under `~/Library/Caches/KiraFrqGen/coreml` (~670 MB for RMVPE), so only the first run
compiles; deleting that folder is safe.

## ML server

A faster machine on the same network can run the ML estimators for another one. Start the server
there:

```
kirafrqgen-cli --serve                  # port 39470; the GPU runner when available
kirafrqgen-cli --serve --token secret   # require a token from clients
```

It prints the addresses to connect to, its ML runner and its ML estimators, and runs until
Ctrl-C. If port 39470 is taken it warns and picks another; `--port N` sets one explicitly. On
macOS the first start asks whether to accept incoming connections; allow it, or clients cannot
connect. RMVPE needs `rmvpe.onnx` on the server; the server's model files are the ones used,
not the client's.

On the client, choose Remote server in the GUI's Advanced settings (host, port, token, and Test
connection), or on the CLI:

```
kirafrqgen-cli bank --ml-runner remote --ml-server 192.168.1.20
kirafrqgen-cli bank --ml-runner remote --ml-server 192.168.1.20:39470 --ml-server-token secret
```

Only ML f0 estimation travels: each wav's audio goes to the server unchanged (lossless) and its
f0 comes back, so the tables match a local run on the server's runner. DIO and Harvest still run
on the client. The token keeps casual users on the network out; the connection is plain HTTP,
not encrypted, so use it on a network you trust.

## License

[MIT](LICENSE)
