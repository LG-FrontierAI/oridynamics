# Five example systems

[five_cases_with_msa.json](five_cases_with_msa.json) contains the five systems displayed in the repository README: three protein monomers, one protein–ligand complex, and one protein–protein assembly, with matching precomputed A3M alignments in [msas/](msas/). Paths are relative to the JSON file, so the examples work after cloning the repository into a new location.

## Single-GPU

Run the commands below from the root of the code checkout.

Follow [INSTALL.md](../INSTALL.md) to install the inference environment and configure `ORIDYNAMICS_PATHS` and `ORIDYNAMICS_ASSETS` for the model weights and CCD resources. MSA search tools and databases are not needed for the bundled examples.

```bash
oridynamics sample \
  --input-json examples/five_cases_with_msa.json \
  --out runs/five_cases
```

This generates 32 final conformations per system, or 160 in total. For each system, PTX uses sixteen seeds (`0–15`) with two candidates each and selects the highest-ranked initial structure; four ensemble seeds then generate eight samples each. PTX evaluates the trunk once per seed and shares its output across that seed's candidates. On the first run, each final ensemble is saved as `runs/five_cases/targets/<name>/ensemble.cif`. If that output directory already exists, a host name, timestamp, and unique suffix are added automatically; the actual output directory is printed at startup. Add `--plan` to validate inputs and resource paths and preview the output path without model inference or output creation.

For a larger ensemble, increase `--samples-per-seed`.

<a id="run-on-multiple-gpus"></a>

## Multi-GPU

Select the GPUs with `--gpus`, for example `0,1` or `0,2,3`. The launcher distributes complete systems across them:

```bash
python -m oridynamics.multi_gpu \
  --input-json examples/five_cases_with_msa.json \
  --gpus 0,1,2,3
```

Each worker uses one GPU and processes complete systems, keeping all partners in a complex together. The launcher approximately balances the targets by protein residue count. Each target still receives 32 PTX candidates and 32 final samples; adding GPUs does not multiply the sample count. A single system is not split across GPUs, and GPU memory is not pooled. The five examples can use up to five workers.

Without `--out`, the launcher creates `runs/multigpu_<host>_<timestamp>_<unique ID>/`. Worker logs are in `logs/gpu_<index>.log`, progress is recorded in `multi_gpu.json`, and final structures are in `gpu_<index>/targets/<name>/ensemble.cif`. An explicit `--out` receives a host/timestamp/unique suffix automatically if it already exists. Code, weights, CCD files, and source MSAs are reused; each worker writes its own outputs and caches. Shared resources do not need to be downloaded again on another machine.

GPU indices refer to the devices exposed by `CUDA_VISIBLE_DEVICES` when it is set. Add `--plan` to preview the assignment and validate input/resource paths without starting workers or creating output directories; this does not check GPU availability or memory capacity.

## Custom sampling options

Both commands default to 32 PTX candidates (sixteen seeds × two), 32 final conformations per system (four seeds × eight), and `--sample-chunk-size 1`. To set these options and the output directory explicitly:

```bash
python -m oridynamics.multi_gpu \
  --input-json examples/five_cases_with_msa.json \
  --gpus 0,2 \
  --anchor-candidates 32 --anchor-samples-per-seed 2 --anchor-seed 0 \
  --seeds 0,1,2,3 --samples-per-seed 8 \
  --sample-chunk-size 4 \
  --out runs/custom
```

The sampling options also work with `oridynamics sample`. `--sample-chunk-size` controls the number of ensemble samples processed together on each GPU. Larger values require more memory; they do not change the final sample count or PTX candidate budget. Increase it when GPU memory allows. Changing chunk size can change individual stochastic samples even with the same seeds.

## Included alignments

| System | Type | Chain | Residues | A3M records, including query |
| :--- | :--- | :--- | ---: | ---: |
| [6odd_B](https://www.rcsb.org/structure/6ODD) | Monomer | B | 74 | 2,285 |
| [7e2s_A](https://www.rcsb.org/structure/7E2S) | Monomer | A | 233 | 1,808 |
| [6l3r_E](https://www.rcsb.org/structure/6L3R) | Monomer | A | 669 | 10,070 |
| [2K0G](https://www.rcsb.org/structure/2K0G) | Protein–ligand, `CCD_CMP` (cAMP) | A | 142 | 4,896 |
| A00C2, DynaRepo-PPI | Protein–protein | A | 381 | 5,219 |
| A00C2, DynaRepo-PPI | Protein–protein | B | 53 | 7,280 |

Chain IDs refer to the bundled JSON inputs. The 2K0G ligand uses chain B.

Every query matches its example sequence exactly, and every aligned row has the required number of columns. All six files are unpaired MSAs; the PP chains have separate alignments, with no supplied paired MSA. Ligands do not require MSAs. Record counts describe the supplied files, not the number of rows retained by model featurization.

Keep the supplied sequences unchanged so that they match the bundled MSAs.

## Other MSA modes

[five_cases.json](five_cases.json) contains the same systems without MSA paths. After the separate Python environment setup in [INSTALL.md](../INSTALL.md#4-local-msa-search-with-colabfoldmmseqs2), search your own local database with:

```bash
oridynamics sample --input-json examples/five_cases.json \
  --msa-local-db /data/colabfold_db --msa-threads 16 \
  --msa-search-executable "$PWD/.envs/msa/bin/colabfold_search" \
  --msa-mmseqs "$PWD/vendor/mmseqs18/mmseqs/bin/mmseqs" \
  --seeds 0,1,2,3 --samples-per-seed 8 --sample-chunk-size 1 \
  --out runs/five_cases_local_search
```

Replace the database and executable paths if your installation uses different locations.

For a small query-only installation test, use one PTX candidate and two final samples per system:

```bash
oridynamics sample --input-json examples/five_cases.json --single-sequence \
  --anchor-candidates 1 \
  --seeds 0 --samples-per-seed 2 --sample-chunk-size 1 \
  --out runs/five_cases_query_only
```

See the [complex-input guide](../docs/complex_inputs.md) for new systems, per-chain MSAs, and optional initial structures.
