# Five example systems

[five_cases_with_msa.json](five_cases_with_msa.json) contains the five systems displayed in the repository README: three protein monomers, one protein–ligand complex, and one protein–protein assembly, with matching precomputed A3M alignments in [msas/](msas/). Paths are relative to the JSON file, so the examples work after cloning the repository into a new location.

## Run the examples

Run the commands below from the root of the code checkout.

Follow [INSTALL.md](../INSTALL.md) to install the inference environment and configure `ORIDYNAMIC_PATHS` and `ORIDYNAMIC_ASSETS` for the model weights and CCD resources. MSA search tools and databases are not needed for the bundled examples.

```bash
oridynamic sample --input-json examples/five_cases_with_msa.json \
  --seeds 0 --samples-per-seed 2 --sample-chunk-size 1 \
  --out runs/five_cases
```

This generates two final conformations per system, or ten in total. Each final ensemble is saved as `runs/five_cases/targets/<name>/ensemble.cif`. Use a new output directory for each run. Add `--plan` to validate inputs and resource paths without model inference or output creation.

For a larger ensemble, increase `--samples-per-seed`, for example to `40`.

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
  --seeds 0 --samples-per-seed 2 --sample-chunk-size 1 \
  --out runs/five_cases_local_search
```

For an explicitly query-only installation test:

```bash
oridynamic sample --input-json examples/five_cases.json --single-sequence \
  --seeds 0 --samples-per-seed 2 --sample-chunk-size 1 \
  --out runs/five_cases_query_only
```

See the [complex-input guide](../docs/complex_inputs.md) for new systems, per-chain MSAs, and optional initial structures.
