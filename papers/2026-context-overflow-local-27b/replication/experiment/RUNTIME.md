# Runtime identities and setup boundaries

This source component is preparation, not a completed experiment or an
approved release. `ENVIRONMENT.json` contains selected, hashed observations;
`ASSETS.json` separately identifies model and dataset files. Neither is an
offline backup. No native libraries, weights, virtualenvs or benchmark text
are included, and no package installation is performed by the checkers.

## What the records establish

| Record | What it establishes | What it does not establish |
|---|---|---|
| P5 pre-primary snapshot | Dated hardware, installed metadata, selected native file hashes and the server's captured environment | The earlier pilot/preprocessing environment, continuous monitoring or completed primary inference |
| EXL3 base and overlay metadata | Distinct installed 1.4.3 and active lab 1.4.7 identities, including the mapped overlay extension | By itself, authenticated wheel acquisition or a working reconstructed dependency stack |
| Separate P5 runtime-evidence records | Hash-pinned upstream wheel, complete byte-only overlay reconstruction and comparison with the dated snapshot | Package installation, native import, model execution or proof every inherited patch was exercised |
| Lingua environment (P5 only) | Currently installed compressor/dependency metadata | Proof those exact library bytes produced previously cached probabilities |
| Analysis environment | Current NumPy/SciPy/Matplotlib version metadata used for the preparation tooling | A complete transitive dependency lock or an inference environment |
| P6 build | Current clean source revision and selected CMake/compiler settings | A model-load measurement, complete native identity or any of the planned 7,200 responses |

The original snapshot's UTC start/end and the metadata collection's UTC date
are separate. Only Name/Version headers are projected from distribution
metadata; source descriptions, contact information, direct URLs and package
index credentials are not copied. A METADATA hash identifies that metadata
file, not all installed package contents or an upstream wheel archive.
Selected P5 native hashes come from the dated snapshot and are explicitly
not freshly rehashed by this collector. `mapped_at_snapshot: false` means
an inventoried file was not observed mapped by that server, not that it
was loaded by a different stage. This is not the entire mapped-library set.

## Keep the environments separate

The EXL3 overlay and Lingua runtime are P5-specific. P6 serves its GGUFs
with llama.cpp; its recorded base Python environment supports harness and
tokenizer work, not EXL3 generation. The complete installed-package inventory
is not a claim that every listed distribution is required for that paper.

The existing EXL3 environment uses CPython 3.12.3 and torch 2.10.0+cu128.
Its site-packages includes stock ExLlamaV3 1.4.3, but P5's `PYTHONPATH`
selects the lab-patched 1.4.7 overlay and its CPython-3.12 native extension.
Use the exact observed versions in the JSON; do not install both EXL3
versions into one site-packages directory and assume precedence is unchanged.

P5's Lingua environment is separate: the inspected installation contains
LLMLingua 0.2.2 and a different torch/transformers stack. Current metadata
is useful for prospective reconstruction, not a retrospective preparation
attestation. Before a new compressor/retrieval pass, record that process's
actual environment and repeat the relevant algorithm-equivalence checks.
Do not merge the two environments or upgrade the live campaign's packages.

The metadata projection's two `serving.patched_sources` entries describe its
selected generator/architecture observations, not every modified file in the
overlay. Their `before` records came from preserved local files. The separate
P5 wheel/reconstruction records below subsequently establish that those input
bytes also match the hash-pinned publisher wheel. Do not relabel the earlier
metadata collector as having performed this later check.

Use the pinned TabbyAPI commit in `ENVIRONMENT.json`, not its generic
editable-install version `0.0.1`. P5 uses `EXL3_CKPT_PP=1024`, a 4,096 MiB
host recurrent stash and the included experiment configuration, not the
16 GiB daily-driver configuration. The captured environment records only
its declared allowlist. P5's source clears the child's `LD_LIBRARY_PATH`;
this is not proof that every possible `LD_*` variable was cleared throughout
the original run. Future guarded launch records remain mandatory.

## P5: exact wheel and three-file overlay reconstruction

The [publisher's v1.4.7 release](https://github.com/turboderp-org/exllamav3/releases/tag/v1.4.7)
contains `exllamav3-1.4.7+cu128.torch2.10.0-cp312-cp312-linux_x86_64.whl`:
298,402,779 bytes, SHA-256
`57bd760ebaccb8247a1c89503489f4390f3f98282ca501d8e87e5bae5b749e42`.
P5's `runtime-evidence/wheel-verification.json` records anonymous HTTPS
publisher metadata readback and verification of the entire archive, all 467
members and their wheel RECORD hashes. This is not an independent signature.
The native extension matches P5's dated snapshot byte-for-byte.

Reconstruction needs **three** changed Python files. The helper applies the
combined generator patch, only the architecture section of
`exl3_spec_sampled.patch`, and the inherited patch under `patches/withdrawn/`.
It uses exact input/output hashes, offsets and context, without running patch
commands or trusting their archival header paths. The withdrawn quantizer
change affects the conversion routine `block_ldl`; it was retained in the
installed overlay after its proposed bug fix was withdrawn. Keeping those
bytes reproduces the observation, not an endorsement, optimization claim or
proof the fallback ran in P5. Experimental sampled-draft/measurement switches
also remain off for this study.

`runtime-evidence/overlay-reconstruction.json` records the 467 reconstructed
files (1,299,734,393 bytes). All **453** members captured by the earlier P5
snapshot match, including the three modified sources. The remaining **14**
small license/text files match a separately dated current readback, not the
earlier snapshot; see `observation-comparison.json`. No native extension or
engine was imported, no model was read and no serving installation was changed.

For a new byte-only reconstruction, first check live machine availability and
choose a host with sufficient CPU/RAM/storage. The destination's parent must
already exist; all output names must be new. From the source component:

```bash
python3 -B verify_p5_runtime_wheel.py /absolute/new-cache/exllamav3.whl --download --receipt /absolute/new-cache/wheel-verification.json
python3 -B prepare_p5_runtime_overlay.py --wheel /absolute/new-cache/exllamav3.whl --patches /absolute/source-component/replication/patches --output /absolute/new-overlay
python3 -B prepare_p5_runtime_overlay.py --verify /absolute/new-overlay
```

Only the explicit `--download` call uses the network. The helpers do not
install dependencies, compile, load native code, download weights or run
inference. A failed partial new overlay is retained and must not be reused
as a successful build. The original wheel RECORD is preserved as historical
input; it does **not** describe the three patched files. The new
`OVERLAY_MANIFEST.json` is the hash inventory for the unused reconstructed
overlay. This is not a new redistributable wheel or `pip`-managed installation.

CPython/Torch/CUDA dependency setup, loader behavior, TabbyAPI, model access
and a new prospective execution baseline remain separate requirements.
Preserve the original working runtime; neither the source component nor
these text receipts are a backup of that complete environment.

## The two llama.cpp builds are not interchangeable

Both use commit `035e22731a7fd70b9854b3a2d64ec68e9b1a45d3`, Release, CUDA
architecture 89 and flash attention enabled. P5's inspected build has
`GGML_CUDA_FA_ALL_QUANTS=OFF`; P6's has `GGML_CUDA_FA_ALL_QUANTS=ON` for
its mixed q8/q4 KV condition. The inspected compiler records identify
GNU C++ 13.3.0 and NVIDIA CUDA 12.6.85. Consult all selected options in the
JSON, including `GGML_NATIVE`, shared-library, OpenMP and backend switches.
Compiler records are current build-tree metadata, not a historical compiler
execution log. Identical switches need not yield byte-identical binaries.

A new build is a heavy job: inspect candidate machines' live CPU, RAM and
GPU headroom first. Build in a separate directory on an appropriate host;
do not rebuild the active engine in place. The scoped CUDA 12.6 runtime
library path belongs to the llama.cpp child. Do not globally inject it into
the separate torch environment. Record actual executable/shared-library
hashes, loader mappings and server build readbacks before new requests.
P6's runner already has those prospective measurement gates; the empty
native-artifact list here is intentional, not a successful binary check.

## Practical reproduction order

1. Run `replication_source.py` and its guarded `--self-test` using an
   analysis Python with NumPy/SciPy installed. The recorded analysis
   versions are NumPy 2.3.3, SciPy 1.16.2 and Matplotlib 3.10.6 at this
   checkpoint. The JSON, not this dated prose, is the machine-readable
   inventory. Scored-data analysis/PDF reconstruction does not need model
   inference; its separate result component will exist only after real
   measurements pass validation.
2. For full inference, obtain the model/data assets at their immutable
   revisions and verify their hashes with `replication_check.py --verify`
   after checking machine availability. Preserve dataset/model license
   restrictions; inclusion of an identity is not redistribution permission.
3. Recreate the separate runtimes above, record path/catalog source
   adaptations and their hashes, and establish a new prospective baseline.
   P6's included `prepare_p6_rerun.py` creates that separate source layout
   and records its narrow adaptations; it does not reconstruct native runtime
   dependencies or run measurements. P5's `prepare_p5_rerun.py` likewise creates
   a private source layout with explicit model/data/cache/runtime paths and
   declared narrow changes. Its complete oracle prompt/date-alignment rebuild
   has been checked. The new workspace's separate `freeze_inputs.py` checks
   datasets/tokenizer, the complete pilot/cap, all prompt cells and source
   identities; explicit `--freeze` creates a private input-consistency record.
   It is not execution evidence. A new guarded baseline/pilot/measurement
   route remains separate work. Do not use its
   original-study wrapper as a foreign-run installer or overwrite its baseline.
   Other installation sources and dependency wheels still need authenticated
   identities; the EXL3 check above does not cover them. The current version
   inventory is not a `pip --require-hashes` lockfile.
4. Use `FULL_RERUN.md` and the included protocol/harness notes to prepare
   prompts, make the new campaign's pilot decision and run measurements.
   Preserve all attempts, timing, placement and dependency evidence.
   Never reuse original private execution receipts as proof of a new run.

No broad automatic setup script is shipped: the source provides experiment
and verification code, while model access, native dependency setup,
hardware choice and a new campaign baseline remain explicit operator work.
Final release integration and findings-specific manuscript review remain.
