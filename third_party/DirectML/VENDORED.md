# DirectML runtime

`DirectML.dll` is not committed to this repository. The pinned Windows `ort`
build downloads Microsoft's DirectML-enabled ONNX Runtime archive from
pyke.io, and `ort-sys`'s `copy-dylibs` extracts the runtime next to the built
executables (`target/<profile>/DirectML.dll`), so local builds and the Windows
release artifacts ship Microsoft's binary unchanged.

- Source archive: `x86_64-pc-windows-msvc+directml.tar.lzma2`, listed in
  `ort-sys`'s `build/download/dist.tsv` (ONNX Runtime 1.28.0); sha256
  `f7c654b3729cb9e5ad2a36a0c38e5b48e63bf4eed22968931aed33a0ad0b527d`.
- Binary: DirectML Redistributable Library 1.15.4
  (`1.15.4+241025-1615.1.dml-1.15.fac7597`); sha256
  `9c9e6d822561c6c41b90e6994b3e8857cf1d66dbfb1e0c4c799c7c89b4e92da1`.

`LICENSE.txt` and `ThirdPartyNotices.txt` are the texts from the
`Microsoft.AI.DirectML` 1.15.4 NuGet package and must accompany
redistributions; the Windows release artifacts copy them next to the DLL as
`DirectML.LICENSE.txt` and `DirectML.ThirdPartyNotices.txt`.
