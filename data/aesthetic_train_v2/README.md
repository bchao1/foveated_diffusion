# Aesthetic-Train-V2 split

Our train/test split of [zhang0jhon/Aesthetic-Train-V2](https://huggingface.co/datasets/zhang0jhon/Aesthetic-Train-V2) (HF revision `3902ab5`). Every foveated and baseline LoRA trains on `metadata_train.csv`. Table 1 and the rebuttal ablations evaluate on `metadata_test.csv`.

| File | Rows | md5 |
|---|---|---|
| `metadata_train.csv` | 95,288 | `d75d59016c0822c89d8d2ddcfa046e6c` |
| `metadata_test.csv` | 10,000 | `26e56af5e053c11542d028fe3f7a31e1` |

Columns: `image,prompt`. `image` is relative to the dataset root (`images/NNNN/<id>.jpg`), which is the layout after you extract the HF tars. These are copies of apollo `/local/brian/Aesthetic-Train-V2/metadata_{train,test}.csv`. Do not regenerate them, because the split is ours.
