"""
where-teacher-acts: the one place that decides which ePillID images train, validate and test a model.

ePillID ships five consumer folds that are disjoint by pill type (192 pill types each). The
benchmark's own script (src/train_cv.py of usuyama/ePillID-benchmark) holds the last fold out as
the test set and lets each other fold serve once as validation. This study follows its first
run: train on folds 1 to 3, validate on fold 0, test on fold 4.

Validation is used exactly as in the earlier studies, by the same trainer code: fold-0
consumer images are the queries, all 9,804 references the gallery, the score is the consumer-side
mAP with both-sides fusion, it is computed every 10 epochs and in each of the last 5, and the
checkpoint is kept whenever it improves. Only two things change: the models never train on fold 4,
and the deployment benchmark reads fold 4. The same holds for the helpers that the deployment
benchmark shares with the earlier study: the evaluation loader takes the test fold as its argument,
and the PTQ calibration pool is every image except the test fold's consumer queries, the rule the
earlier study used.

Training sees the consumer images of folds 1 to 3 and the references of their pill types only, so
validation and test pill types are unseen (zero-shot). This is stricter than the benchmark's own
script, which trains on every reference.
"""
TRAIN_FOLDS = (1, 2, 3)
VAL_FOLD = 0
TEST_FOLD = 4

EXPECTED = {"train_consumer": 2237, "train_ref": 1152, "train": 3389, "val": 746, "test": 745,
            "gallery": 9804, "train_pill_types": 576, "val_pill_types": 192, "test_pill_types": 192}


class Split:
    def __init__(self, df_all, train, val, test, gallery):
        self.df_all, self.train, self.val, self.test, self.gallery = df_all, train, val, test, gallery
        self.num_sub_classes = int(df_all['sub_label_idx'].nunique())

    def counts(self):
        tr = self.train
        return {"train_consumer": int((tr.is_ref == 0).sum()), "train_ref": int((tr.is_ref == 1).sum()),
                "train": len(tr), "val": len(self.val), "test": len(self.test), "gallery": len(self.gallery),
                "train_pill_types": int(tr.label_idx.nunique()), "val_pill_types": int(self.val.label_idx.nunique()),
                "test_pill_types": int(self.test.label_idx.nunique())}


def load_split(check=True):
    """Train, validation, test and gallery frames, built the way the inherited trainers build theirs:
    boolean masks over the full table, so row order (and with it the sampler's draw order under
    a fixed seed) follows the same convention."""
    from utils.data_utils import load_epill_full_data
    df_all = load_epill_full_data()
    consumer = df_all['is_ref'] == 0
    cond_train = consumer & df_all['fold'].isin(TRAIN_FOLDS)
    train_labels = df_all[cond_train]['label_idx'].unique()
    cond_ref_train = (df_all['is_ref'] == 1) & (df_all['label_idx'].isin(train_labels))
    split = Split(
        df_all=df_all,
        train=df_all[cond_train | cond_ref_train].reset_index(drop=True),
        val=df_all[consumer & (df_all['fold'] == VAL_FOLD)].reset_index(drop=True),
        test=df_all[consumer & (df_all['fold'] == TEST_FOLD)].reset_index(drop=True),
        gallery=df_all[df_all['is_ref'] == 1].reset_index(drop=True),
    )
    if check:
        problems = check_split(split)
        if problems:
            raise RuntimeError("where-teacher-acts split check failed: " + "; ".join(problems))
    return split


def check_split(split):
    """Return a list of problems (empty when the split is what the design says)."""
    problems = []
    got = split.counts()
    for k, v in EXPECTED.items():
        if got[k] != v:
            problems.append(f"{k} = {got[k]}, expected {v}")
    tr_types = set(split.train.label_idx)
    va_types, te_types = set(split.val.label_idx), set(split.test.label_idx)
    if tr_types & va_types:
        problems.append(f"{len(tr_types & va_types)} validation pill types also in training")
    if tr_types & te_types:
        problems.append(f"{len(tr_types & te_types)} test pill types also in training")
    if va_types & te_types:
        problems.append(f"{len(va_types & te_types)} pill types shared by validation and test")
    tr_paths = set(split.train.image_path)
    if tr_paths & set(split.val.image_path):
        problems.append("validation images in training")
    if tr_paths & set(split.test.image_path):
        problems.append("test images in training")
    if set(split.train[split.train.is_ref == 0].fold) - set(TRAIN_FOLDS):
        problems.append("training consumer images from a non-training fold")
    return problems


def calibration_pool(df_all, test_fold=TEST_FOLD):
    """The frame the inherited calibration builder samples from: every image except the test fold's
    consumer queries (journal2_pipeline._quant_utils.build_calibration_loader, fold_excluded)."""
    return df_all[df_all['fold'] != test_fold]


def calibration_loader(img_size, exclude_folds, seed, num_samples, batch_size=8):
    """The inherited calibration builder with a set of excluded consumer folds instead of one: the same
    sampling (pandas .sample, random_state=seed) and the same transform. For the test benchmark,
    exclude_folds={TEST_FOLD} reproduces build_calibration_loader(fold_excluded=TEST_FOLD) exactly
    (checked by 00_sanity_split). Decisions scored on validation exclude {VAL_FOLD, TEST_FOLD}, so no
    validation query calibrates the model it scores and no test image is touched at all."""
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from utils.data_utils import load_epill_full_data
    from utils.dataset_loader import PillDataset
    df_all = load_epill_full_data()
    pool = df_all[~df_all['fold'].isin(list(exclude_folds))].reset_index(drop=True)
    if len(pool) > num_samples:
        pool = pool.sample(n=num_samples, random_state=seed).reset_index(drop=True)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return DataLoader(PillDataset(pool, transform=transform), batch_size=batch_size, shuffle=False, num_workers=0)


def _selftest():
    s = load_split(check=False)
    problems = check_split(s)
    print(s.counts())
    print("OK" if not problems else problems)


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _selftest()
