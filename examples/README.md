# Five example systems

[five_cases_with_msa.json](five_cases_with_msa.json) contains the five systems displayed in the repository README: three protein monomers, one protein–ligand complex, and one protein–protein assembly, with matching precomputed A3M alignments in [msas/](msas/). Paths are relative to the JSON file, so the examples work after cloning the repository into a new location.

## Run the examples

Run the commands below from the root of the code checkout.

Follow [INSTALL.md](../INSTALL.md) to install the inference environment and configure `ORIDYNAMIC_PATHS` and `ORIDYNAMIC_ASSETS` for the model weights and CCD resources. MSA search tools and databases are not needed for the bundled examples.

```bash
oridynamic sample --input-json examples/five_cases_with_msa.json \
  --anchor-candidates 32 \
  --seeds 0,1,2,3 --samples-per-seed 8 --sample-chunk-size 1 \
  --out runs/five_cases
```

This generates 32 final conformations per system, or 160 in total. For each system, PTX uses four seeds with eight candidates each and selects the highest-ranked initial structure; four ensemble seeds then generate eight samples each. PTX evaluates the trunk once per seed and shares its output across that seed's candidates. Each final ensemble is saved as `runs/five_cases/targets/<name>/ensemble.cif`. Use a new output directory for each run. Add `--plan` to validate inputs and resource paths without model inference or output creation.

For a larger ensemble, increase `--samples-per-seed`.

## Run on multiple GPUs

The standard `oridynamic sample` command uses one GPU. To distribute the examples across four GPUs, use:

```bash
python -m oridynamic.multi_gpu \
  --input-json examples/five_cases_with_msa.json \
  --gpus 0,1,2,3 \
  --anchor-candidates 32 --anchor-samples-per-seed 8 \
  --seeds 0,1,2,3 --samples-per-seed 8 \
  --sample-chunk-size 2
```

Each worker uses one GPU and processes complete systems, keeping all partners in a complex together. The launcher approximately balances the targets by protein residue count. Each target still receives 32 PTX candidates and 32 final samples; adding GPUs does not multiply the sample count. A single system is not split across GPUs, and GPU memory is not pooled. The five examples can use up to five workers.

Without `--out`, the launcher creates `runs/multigpu_<host>_<timestamp>_<unique ID>/`. Worker logs are in `logs/gpu_<index>.log`, progress is recorded in `multi_gpu.json`, and final structures are in `gpu_<index>/targets/<name>/ensemble.cif`. An explicit `--out` must name a new directory. Code, weights, CCD files, and source MSAs are reused; each worker writes its own outputs and caches. Shared resources do not need to be downloaded again on another machine.

GPU indices refer to the devices exposed by `CUDA_VISIBLE_DEVICES` when it is set. Add `--plan` to preview the assignment and validate input/resource paths without starting workers or creating output directories; this does not check GPU availability or memory capacity.

`--sample-chunk-size` controls the number of ensemble samples processed together on each GPU. Larger values can improve throughput and require more memory; they do not change the final sample count or PTX candidate budget. The command above uses 2; use 1 if memory is insufficient. Changing chunk size can change individual stochastic samples even with the same seeds. H20 memory capacity and speed must be checked on the actual system.

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

[five_cases.json](five_cases.json) contains the same systems without MSA paths. To search your own local database:

```bash
oridynamic sample --input-json examples/five_cases.json \
  --msa-local-db /data/colabfold_db --msa-threads 16 \
  --seeds 0,1,2,3 --samples-per-seed 8 --sample-chunk-size 1 \
  --out runs/five_cases_local_search
```

For a small query-only installation test, use one PTX candidate and two final samples per system:

```bash
oridynamic sample --input-json examples/five_cases.json --single-sequence \
  --anchor-candidates 1 \
  --seeds 0 --samples-per-seed 2 --sample-chunk-size 1 \
  --out runs/five_cases_query_only
```

See the [complex-input guide](../docs/complex_inputs.md) for new systems, per-chain MSAs, and optional initial structures.
