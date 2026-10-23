# ML server: only exact ML estimation crosses the network

Status: accepted

The remote ML runner (#85) splits a run between a client and an **ML server**. The client keeps
everything voicebank-shaped: the scan, WAV decode, the DSP estimators, the energy voicing gate,
table building, every writer and the mrq code-page handling. The server runs one thing: an ML
estimator over one wav's audio, answering with f0 on the 5.805 ms table grid. It holds no
voicebank state, so a request carries all it needs and any client can use any server. The
remote estimator sits behind core's `F0Estimator` seam, so the pipeline has no remote branch.

The audio crosses the wire exactly as the client decoded it: mono 44.1 kHz, as 16-bit FLAC
when every sample is on the 16-bit grid (a normal UTAU wav), 24-bit FLAC on the 24-bit grid,
else raw f64. The server runs the same 44.1 kHz entry a local run does, so on the same runner
and thread count a remote table equals a local one bit for bit; any difference comes from the
server's runner, never from the transfer.

## Considered options

- **16 kHz, 16-bit FLAC (the model rate)** — about a third of the transfer (the exact wire is
  about 700 kbps of PCM per stream before FLAC, small next to compute on a LAN), but lossy:
  resampling to 16 kHz moves samples off the 16-bit grid, and rounding them back is enough to
  flip SwiftF0's voicing at quiet onsets (measured). 24-bit at 16 kHz still flips frames right
  after digital silence. Downsampling or rounding on the wire is rejected; so is any lossy
  codec (Opus).
- **A pool of parallel model sessions on the server** — rejected; the server keeps one session
  per ML estimator and requests queue on it, as estimates do in a local run.

## Consequences

- The server's model files win: its `rmvpe.onnx`, its `swiftf0.onnx` override or bundled copy,
  and its default confidence threshold. A client's local model override does not apply to
  remote runs.
- Compatibility is the protocol version (`/v1`, reported by `GET /v1/info`): a mismatch stops
  the run before any file. The KiraFrqGen version never enters the check, not even as a warning;
  it only names the releases in a mismatch error. So a change that alters what the server
  computes for a request bumps the protocol version.
- A failed request is retried once, then fails that wav. There is never a fallback to local
  estimation, so the user always knows which machine produced a table.
