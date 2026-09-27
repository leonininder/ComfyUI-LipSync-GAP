# Vendored source and license provenance

Checked 2026-09-27. This is an evidence inventory, not a blanket license grant. The inherited README explicitly declares Apache 2.0. Its exact declaration is retained; missing a separate LICENSE file alone does not negate that declaration. It cannot grant permission for third-party material beyond the author's rights.

| Local component | Source evidence | Observed upstream terms / remaining boundary |
|---|---|---|
| `nodes.py`, integration and workflow | [GeekatplayStudio/ComfyUI-LipSync-GAP](https://github.com/GeekatplayStudio/ComfyUI-LipSync-GAP) | Explicit README declaration: Apache 2.0 (see LICENSE files where applicable). Preserved at exact inherited commit in `licenses/geekatplay-readme-declaration.md`; full Apache text retained. This is affirmative evidence for its own integration code, not blanket third-party clearance. |
| ByteDance-header files under `latentsync/` | [LatentSync LICENSE](https://github.com/bytedance/LatentSync/blob/main/LICENSE) and local per-file Apache headers | Apache 2.0 source text saved below; this does not automatically cover adaptations from every other project. |
| `latentsync/whisper/whisper/` | [OpenAI Whisper LICENSE](https://github.com/openai/whisper/blob/main/LICENSE) | MIT source notice retained in `licenses/openai-whisper.txt`. Modified vendored code is not claimed byte-identical to upstream. |
| `latentsync/whisper/audio2feature.py` | Header points to [MuseTalk](https://github.com/TMElyralab/MuseTalk/blob/main/LICENSE) | Composite license text saved in full; GitHub reports NOASSERTION. Review the actual text and relevant component, not the repository badge alone. |
| pipeline and motion module | Headers point to [AnimateDiff](https://github.com/guoyww/AnimateDiff/blob/main/LICENSE.txt) | Apache 2.0 upstream text saved; adapted files retain attribution. |
| attention/resnet implementations | Headers point to [Diffusers](https://github.com/huggingface/diffusers/blob/main/LICENSE) | Apache 2.0 source text saved. |
| `latentsync/trepa/third_party/VideoMAEv2/` | [VideoMAEv2](https://github.com/OpenGVLab/VideoMAEv2/blob/master/LICENSE) | MIT upstream text saved; headers also reference BEiT/timm/DINO/DeiT, which still require file-level provenance mapping. Training-only subtree removed from current working tree; historical source remains recorded in Git and removal manifest. |
| `latentsync/utils/audio.py` | Header points to [Wav2Lip](https://github.com/Rudrabha/Wav2Lip#disclaimer) | Upstream explicitly limits repository use to personal/research/non-commercial purposes. This helper and its only two training-dataset callers were removed from the current working tree after import search. Historical commits retain their original terms. |
| `latentsync/utils/affine_transform.py` | Header points to [StyleSync](https://github.com/guanjz20/StyleSync) | Inherited adapter removed and replaced with an OpenCV API implementation. Current code has new three-anchor affine solving, anchor smoothing and CPU feathered compositing; crop dimensions/anchors retain the model input contract. This is not a clean-room claim or historical permission grant. Geometry differs from the old similarity fit and needs visual quality evaluation. |
| `latentsync/trepa/utils/metric_utils.py` | Header points to [stylegan-v](https://github.com/universome/stylegan-v) | Training-only subtree removed; historical source and terms remain documented. |

## Evidence files

- `license-sources.json`: original URL, Git blob identity and downloaded notice SHA-256 for six upstream license texts under `licenses/`.
- `vendored-provenance.json`: local file SHA-256 and source-header mapping. Generated from actual file headers, not inferred ownership. This is a bounded inventory, not a full transitive dependency SBOM.
- Per-file notices in retained inherited files are preserved; `removed-training-files.json` records 13 removed training files and original hashes. The saved notices are upstream source documents; their presence does not retroactively grant rights missing from a different component.

## Model and fixture boundary

[ByteDance LatentSync-1.6 model card](https://huggingface.co/ByteDance/LatentSync-1.6) labels weights OpenRAIL++; source Apache licensing does not replace model terms. `assets-manifest.json` pins 1.6 checkpoint revision and SHA-256. Whisper tiny uses the official checkpoint hash. The MediaPipe landmarker is pinned to Google's versioned storage object and measured hash; its applicable model terms still require a complete redistributable-asset review before bundling. We do not commit checkpoints into Git.

The optional GPU smoke uses the [scikit-image astronaut fixture](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut), identified by its maintainers as a NASA image in the public domain, plus a synthesized sine tone. It is an altered static-image test, not a real recording, speech sample, endorsement by the pictured person, or lip-sync quality demonstration.

## Closure work

The current inference distribution no longer includes Wav2Lip training helpers, TREPA/VideoMAEv2/stylegan-v training modules or the former StyleSync adapter. These removals do not erase Git history or retroactively license it. The inherited author's explicit Apache declaration is retained as evidence for integration code. Remaining work includes complete third-party dependency and model-specific term review and visual regression evidence for the new geometry. No permission request was sent and no third-party license was fabricated.

## OpenCV replacement evidence

The replacement uses documented [getAffineTransform, warpAffine and invertAffineTransform](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html), with numerical anchor, inverse, degeneracy, smoothing and outside-crop preservation tests. Three-point affine fitting can shear a face, unlike the prior similarity fit; a successful GPU smoke does not establish equivalent visual quality. The public interface and crop coordinate contract are preserved. Implementation is original to this change after inspection of the old interface; no clean-room claim is made.
