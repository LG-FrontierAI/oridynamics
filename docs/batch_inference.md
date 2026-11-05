# Batch inference

FASTA and request manifests describe independent protein targets. Use `--input-json` for Protenix-format monomer, protein–ligand, and protein–protein tasks; every entity within a task is generated together. See the [complex-input guide](complex_inputs.md).

## Inputs and execution

Complete the [installation and resource setup](../INSTALL.md) first. Run from the code checkout with the environment activated and `ORIDYNAMIC_PATHS` / `ORIDYNAMIC_ASSETS` set:

```bash
python -B -u -m oridynamic \
  --fasta sequences.fasta --msa-dir msas \
  --seeds 0,1,2,3 --samples-per-seed 8 --sample-chunk-size 1 \
  --out runs/batch_v1
```

The first word of each FASTA header is its unique target name. Alignments are read from `msas/<target>.a3m`. Add `--structure-dir anchors` to provide matching `.cif`, `.mmcif`, or `.pdb` files. Targets without a supplied structure receive a predicted anchor.

An initial structure does not replace the MSA or chemical reference features (`ref_pos`). It serves as the decoder anchor, rather than a trunk template. Its sequence and required heavy atoms must match the input; missing residues or atoms produce an error.

Use `--manifest request.json` to specify custom paths and chain IDs. Relative paths are resolved from the JSON directory:

```json
{
  "schema": "oridynamic_batch_request_v1",
  "targets": [
    {"name": "protein_a", "sequence": "ACDEFG", "msa_a3m": "msas/a.a3m"},
    {"name": "protein_b", "sequence": "ACDEFGH", "initial_structure": "anchors/b.cif", "structure_chain": "A"}
  ]
}
```

The second target has no MSA, so supply a local database as described below.

`--limit 2` restricts processing to the first two targets. `--plan` checks input and resource paths without searching, importing Torch, creating files, or running the model.

## Local and online MSAs

With the database and ColabFold/MMseqs2 tools installed, search and inference can be combined:

```bash
python -B -u -m oridynamic \
  --fasta sequences.fasta --msa-local-db /data/colabfold_db \
  --msa-search-executable /opt/colabfold/bin/colabfold_search \
  --msa-mmseqs /opt/mmseqs/bin/mmseqs --msa-threads 32 \
  --seeds 0,1,2,3 --samples-per-seed 8 --out runs/local_batch_v1
```

Local search defaults to CPU. Add `--msa-device gpu` to use a compatible GPU-enabled ColabFold/MMseqs2 installation; GPU visibility follows `CUDA_VISIBLE_DEVICES`.

You can also supply `--msa-dir`: existing alignments are reused, and only missing ones are searched. The wrapper invokes the search tool separately for each protein entity; it does not guarantee a single database load. Search completes before model loading. Temporary files stay within the new run directory, and public databases are read-only. A failed search stops the run.

See [INSTALL.md](../INSTALL.md) for local installation, CPU/GPU search, and the separate search container. Online search is a separate preparation step described in the [online MSA guide](msa_online.md). `--single-sequence` is an explicit query-only smoke-test option, not a substitute for MSA-based quality evaluation.

## Model lifecycle and memory

1. Validate inputs and prepare all MSAs.
2. Load the cofolding backbone once, prepare each target, and predict or map its anchor.
3. Load the autoencoder, flow module, refine module, and trunk LoRA parameters.
4. Process targets in turn, computing the online LoRA context and generating samples with the selected anchor.

All target anchors are prepared before the joint-model LoRA parameters are applied. By default, PTX generates 32 anchor candidates using seeds 0–15 with two samples per seed. Each seed evaluates the trunk once; its diffusion samples share that context. The native ranking score selects one candidate across all groups, and all final samples use that anchor. `--anchor-candidates` controls the total budget and `--anchor-samples-per-seed` controls the number of candidates sharing each trunk evaluation (default 2); the last group is smaller if needed. Supplying an initial structure skips candidate generation.

Models remain in the process, with CPU/GPU offloading between phases and intermediate tensors released between targets. The defaults `--sample-chunk-size 1` and `--attention-chunk-size 32` limit working memory, but do not guarantee support for arbitrary lengths on smaller GPUs. The pair-context tensors in each run-specific `prepared.pt` grow quadratically with token count, so reserve output-disk space accordingly. They are not a shared cache across runs.

Additional controls include `--anchor-seed` (first PTX seed, default 0; incremented once per candidate group), `--device` (default `cuda:0`), `--dtype` (`bf16` or `fp32`, default `bf16` for trunk/codec/DiT), and `--attention-backend` (`torch` by default; `cuequivariance` requires separately installed compatible extensions). These do not replace the checkpoint-bound flow/refiner sampling configuration.

For multiple GPUs, use the [multi-GPU launcher](../examples/README.md#run-on-multiple-gpus) with JSON inputs and precomputed MSAs. It starts one standard sampler per GPU and distributes complete targets across them. Each process loads its own models; the original single-GPU sampler and per-target sample budget are retained.

## Outputs and timing

All paths below are relative to the directory passed to `--out`.

| Path | Contents |
| :--- | :--- |
| `request.json`, `progress.json` | Batch request and phase/target progress |
| `summary.json`, `timing.json` | Batch completion summary, model-load counts, and timing |
| `failure.json` | Failure details when an error occurs after run initialization; absent after success |
| `targets/<name>/anchor.cif` | Selected initial structure |
| `targets/<name>/ensemble.cif` | Final conformations from all requested seeds |
| `targets/<name>/prepared.pt`, `targets/<name>/prepared.json` | Prepared inputs, full-precision anchor, context tensors, identities, and integrity metadata |
| `targets/<name>/seed_<seed>.pt` | Final coordinate tensor, seed, and atom identities |
| `targets/<name>/input.json`, `targets/<name>/resolved_config.json` | Effective native molecular input and model configuration |
| `targets/<name>/run_metadata.json`, `targets/<name>/completion.json` | Target provenance, sampling settings, and completion status |
| `targets/<name>/anchor_phase/timing.json`, `targets/<name>/ensemble_phase/timing.json` | Per-target phase timings |

For JSON inputs, protein queries and alignments are stored under `targets/<name>/chains/<first_entity_chain_id>/`: `query.fasta`, copied `provided.a3m`/`paired.a3m`, or query-only `query.a3m`, depending on the selected MSA mode. Local search creates `msa/unpaired.a3m` and `msa/search.json` there. For FASTA/manifest inputs, the same files are directly under `targets/<name>/`. Homomeric copies of one protein entity share its alignment. `input_fingerprints.json` is written within the target directory only with `--trace-inputs`.

Each `ensemble.cif` combines samples from all requested seeds. The defaults `--seeds 0,1,2,3 --samples-per-seed 8` produce 32 final conformations per target. Chunk size controls how many samples are processed together without changing the total count. Decoded proposals remain internal to the flow-to-refinement pipeline and are not exported.

Overall wall time includes local MSA search. `inference_wall_seconds_excluding_msa` starts after search and includes initialization, PTX anchor prediction, online trunk computation, ensemble sampling, and output writing. `ensemble_generation` only sums the flow/decoder/refiner sampling intervals; it excludes PTX anchor prediction and online trunk computation. Each `ptx_anchor` timing entry represents one seed's candidate group. Per-target timing excludes time spent waiting for other targets. `ensemble_seed` is nested within `ensemble_target`; do not add both when aggregating elapsed time.

You can reuse the same `--out` on subsequent runs. If the requested path already exists, the runner creates a sibling directory named `<requested>_<host>_<timestamp>_<unique ID>` and prints the actual path at startup. Atomic directory creation also handles simultaneous launches from a shared checkout. Existing results are preserved; each invocation starts a new run. `request.json` records the requested and actual output paths. `--plan` previews a path without reserving or creating it, so use the path printed by the actual run. Existing anchors can be explicitly reused through a new input manifest. The multi-GPU launcher follows the same rule for explicit `--out`, or generates an automatic directory when it is omitted; `multi_gpu.json` records the actual root and each worker's output path.
