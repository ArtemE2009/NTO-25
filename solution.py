import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from catboost import CatBoostRanker, Pool

class C:
    TRAIN = "train.csv"
    TARGETS = "targets.csv"
    CAND = "candidates.csv"
    USERS = "users.csv"
    BOOKS = "books.csv"
    BG = "book_genres.csv"
    DESC = "book_descriptions.csv"
    SUB = "submission.csv"
    VEC = "tfidf.pkl"
    SVD = "svd.pkl"
    PROC = "proc.parquet"
    MODEL = "cb.cbm"
    SCHEMA = "schema.json"

    UID = "user_id"
    BID = "book_id"
    READ = "has_read"
    REL = "relevance"
    SRC = "source"
    TS = "timestamp"

    LIST = "book_id_list"

    U_M = "u_mean"
    U_C = "u_cnt"
    B_M = "b_mean"
    B_C = "b_cnt"
    A_M = "a_mean"
    G_C = "g_cnt"

    GENDER = "gender"
    AGE = "age"
    AID = "author_id"
    YEAR = "publication_year"
    LANG = "language"
    PUB = "publisher"
    AVG = "avg_rating"
    DESC_TXT = "description"

    VAL_T = "train"
    MISS = "-1"
    K = 20

class CFG:
    ROOT = Path.cwd()
    CSV = ROOT / "CSV"
    RAW = CSV
    OUT = ROOT / "output"
    PROC = ROOT / "data"
    PROC.mkdir(exist_ok=True, parents=True)
    OUT.mkdir(exist_ok=True, parents=True)
    MDL = OUT / "models"
    SUB = OUT / "submissions"
    MDL.mkdir(exist_ok=True, parents=True)
    SUB.mkdir(exist_ok=True, parents=True)

    RND = 42
    SPLIT = 0.8
    STOP = 300

    TFMAX = 3000
    TFMIN = 2
    TFMAXDF = 0.95
    TFNGRAM = (1, 3)
    TFSUBLINEAR = True
    SVD_DIM = 128

    CB = {
        "loss_function": "YetiRankPairwise",
        "iterations": 3000,
        "learning_rate": 0.04,
        "depth": 8,
        "l2_leaf_reg": 8,
        "random_seed": RND,
        "verbose": 100,
        "random_strength": 1.0,
        "bagging_temperature": 1.0,
        "rsm": 0.9,
        "eval_metric": f"NDCG:top={C.K}",

    }


def read(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep=";")


def expand(cdf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in cdf.iterrows():
        s = r.get(C.LIST, None)
        if pd.isna(s) or s == "":
            continue
        for b in str(s).split(','):
            rows.append({C.UID: int(r[C.UID]), C.BID: int(b)})
    return pd.DataFrame(rows, dtype="int32")


def load():
    tr = pd.read_csv(CFG.RAW / C.TRAIN,
                     dtype={C.UID: "int32", C.BID: "int32", C.READ: "int8"},
                     parse_dates=[C.TS])
    tr[C.REL] = tr[C.READ].map({1: 2, 0: 1}).astype("int8")
    tr[C.SRC] = C.VAL_T

    users = read(CFG.RAW / C.USERS)
    books = read(CFG.RAW / C.BOOKS).drop_duplicates(C.BID)
    bg = read(CFG.RAW / C.BG)
    desc = read(CFG.RAW / C.DESC)

    tr = tr.merge(users, on=C.UID, how="left").merge(books, on=C.BID, how="left")

    targ = read(CFG.RAW / C.TARGETS)
    cand = read(CFG.RAW / C.CAND)
    return tr, targ, cand, bg, desc


def add_agg(df: pd.DataFrame, tr: pd.DataFrame) -> pd.DataFrame:
    u = tr.groupby(C.UID)[C.REL].agg(['mean', 'count']).reset_index()
    u.columns = [C.UID, C.U_M, C.U_C]
    b = tr.groupby(C.BID)[C.REL].agg(['mean', 'count']).reset_index()
    b.columns = [C.BID, C.B_M, C.B_C]
    a = tr.groupby(C.AID)[C.REL].mean().reset_index()
    a.columns = [C.AID, C.A_M]
    return df.merge(u, on=C.UID, how='left').merge(b, on=C.BID, how='left').merge(a, on=C.AID, how='left')


def add_genre(df: pd.DataFrame, bg: pd.DataFrame) -> pd.DataFrame:
    cnt = bg.groupby(C.BID).size().reset_index(name=C.G_C)
    return df.merge(cnt, on=C.BID, how='left')


def _vec_path() -> Path:
    return CFG.MDL / C.VEC


def _svd_path() -> Path:
    return CFG.MDL / C.SVD


def add_tfidf_train(df: pd.DataFrame, tr: pd.DataFrame, desc: pd.DataFrame) -> pd.DataFrame:
    CFG.MDL.mkdir(exist_ok=True, parents=True)

    train_desc = desc[desc[C.BID].isin(tr[C.BID])].copy()
    train_desc[C.DESC_TXT] = train_desc[C.DESC_TXT].fillna('')

    vec = TfidfVectorizer(
        max_features=CFG.TFMAX,
        min_df=CFG.TFMIN,
        max_df=CFG.TFMAXDF,
        ngram_range=CFG.TFNGRAM,
        sublinear_tf=CFG.TFSUBLINEAR,
    ).fit(train_desc[C.DESC_TXT])
    joblib.dump(vec, _vec_path())

    desc_map = dict(zip(desc[C.BID], desc[C.DESC_TXT].fillna('')))
    mat = vec.transform(df[C.BID].map(desc_map).fillna(''))

    if CFG.SVD_DIM and CFG.SVD_DIM > 0:
        svd = TruncatedSVD(n_components=CFG.SVD_DIM, random_state=CFG.RND)
        mat_red = svd.fit_transform(mat)
        joblib.dump(svd, _svd_path())
        cols = [f'tfsvd_{i}' for i in range(mat_red.shape[1])]
        tf = pd.DataFrame(mat_red.astype('float32'), columns=cols, index=df.index)
    else:
        cols = [f'tf_{i}' for i in range(mat.shape[1])]
        tf = pd.DataFrame(mat.toarray().astype('float32'), columns=cols, index=df.index)

    return pd.concat([df.reset_index(drop=True), tf.reset_index(drop=True)], axis=1)


def add_tfidf_infer(df: pd.DataFrame, desc: pd.DataFrame) -> pd.DataFrame:
    vec: TfidfVectorizer = joblib.load(_vec_path())
    desc_map = dict(zip(desc[C.BID], desc[C.DESC_TXT].fillna('')))
    mat = vec.transform(df[C.BID].map(desc_map).fillna(''))

    if _svd_path().exists():
        svd: TruncatedSVD = joblib.load(_svd_path())
        mat = svd.transform(mat)
        cols = [f'tfsvd_{i}' for i in range(mat.shape[1])]
        tf = pd.DataFrame(mat.astype('float32'), columns=cols, index=df.index)
    else:
        cols = [f'tf_{i}' for i in range(mat.shape[1])]
        tf = pd.DataFrame(mat.toarray().astype('float32'), columns=cols, index=df.index)

    return pd.concat([df.reset_index(drop=True), tf.reset_index(drop=True)], axis=1)


def fill(df: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    g = ref[C.REL].mean()
    for c in [C.U_M, C.B_M, C.A_M]:
        if c in df.columns:
            df[c] = df[c].fillna(g)
    for c in [C.U_C, C.B_C, C.G_C]:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    if C.AGE in df.columns:
        if df[C.AGE].dtype.kind in 'biu':
            df[C.AGE] = df[C.AGE].fillna(df[C.AGE].median())
        else:
            df[C.AGE] = pd.to_numeric(df[C.AGE], errors='coerce').fillna(
                pd.to_numeric(df[C.AGE], errors='coerce').median()
            )

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(C.MISS).astype(str)

    return df


def _get_feature_list(df: pd.DataFrame) -> list[str]:
    drop_cols = {C.SRC, C.TS, C.REL, C.READ}
    feats = [c for c in df.columns if c not in drop_cols]
    leak_cols = {"FI"}
    feats = [c for c in feats if c not in leak_cols]
    return feats


def _cat_base() -> set:
    return {C.UID, C.BID, C.GENDER, C.AID, C.LANG, C.PUB}


def _infer_schema(df: pd.DataFrame, feats: list[str]) -> dict:
    schema = {}
    base_cat = _cat_base()
    for f in feats:
        is_cat = (df[f].dtype == 'object') or (f in base_cat)
        schema[f] = 'cat' if is_cat else 'num'
    return schema


def _save_schema(schema: dict):
    (CFG.MDL / C.SCHEMA).write_text(json.dumps(schema))


def _load_schema() -> dict:
    return json.loads((CFG.MDL / C.SCHEMA).read_text())


def _cat_idx(feats: list[str], schema: dict) -> list[int]:
    return [i for i, f in enumerate(feats) if schema.get(f) == 'cat']


def _enforce_types(df: pd.DataFrame, feats: list[str], schema: dict) -> pd.DataFrame:
    for f in feats:
        if schema.get(f) == 'cat':
            df[f] = df[f].astype(str)
        else:
            df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0).astype('float32')
    return df

# ---------- pipeline ----------

def build_train(tr: pd.DataFrame, bg: pd.DataFrame, desc: pd.DataFrame) -> pd.DataFrame:
    df = tr.copy()
    df = add_genre(df, bg)
    df = add_tfidf_train(df, tr, desc)
    df = fill(df, tr)
    return df


def prepare():
    tr, _, _, bg, desc = load()
    proc = build_train(tr, bg, desc)
    proc.to_parquet(CFG.PROC / C.PROC, index=False, engine='pyarrow')


def _user_temporal_split_strict(df: pd.DataFrame, split: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values([C.UID, C.TS])
    tr_parts = []
    va_parts = []
    for uid, g in df.groupby(C.UID):
        n = len(g)
        if n < 3:
            tr_parts.append(g)
            continue
        val_size = max(2, int(np.ceil(n * (1 - split))))
        tr_size = n - val_size
        if tr_size < 1:
            tr_size = n - 2
            val_size = 2
        tr_parts.append(g.iloc[:tr_size])
        va_parts.append(g.iloc[tr_size:])
    tr_df = pd.concat(tr_parts).reset_index(drop=True)
    va_df = pd.concat(va_parts).reset_index(drop=True) if va_parts else pd.DataFrame(columns=df.columns)


    fixed = []
    for uid, gva in va_df.groupby(C.UID):
        if len(gva) < 2:
            gtr = tr_df[tr_df[C.UID] == uid]
            if len(gtr) > 1:
                moved = gtr.tail(1)
                tr_df = tr_df.drop(moved.index)
                va_df = pd.concat([va_df, moved])
                fixed.append(uid)
        else:
            if gva[C.REL].nunique() < 2:
                gtr = tr_df[tr_df[C.UID] == uid]
                if gtr[C.REL].nunique() > 1:
                    target_rel = gva[C.REL].iloc[0]
                    cand = gtr[gtr[C.REL] != target_rel]
                    if len(cand) > 0:
                        moved = cand.tail(1)
                        tr_df = tr_df.drop(moved.index)
                        va_df = pd.concat([va_df, moved])
                        fixed.append(uid)
    va_df = va_df.sort_values([C.UID, C.TS]).reset_index(drop=True)
    tr_df = tr_df.sort_values([C.UID, C.TS]).reset_index(drop=True)
    return tr_df, va_df


def train():
    if not (CFG.PROC / C.PROC).exists():
        prepare()

    df = pd.read_parquet(CFG.PROC / C.PROC, engine='pyarrow')
    df_t = df[df[C.SRC] == C.VAL_T].copy()
    df_t[C.TS] = pd.to_datetime(df_t[C.TS])

    tr_df, va_df = _user_temporal_split_strict(df_t, CFG.SPLIT)
    tr_df = add_agg(tr_df, tr_df)
    va_df = add_agg(va_df, tr_df)
    tr_df = fill(tr_df, tr_df)
    va_df = fill(va_df, tr_df)
    feats = _get_feature_list(tr_df)
    schema = _infer_schema(tr_df, feats)
    tr_df = _enforce_types(tr_df, feats, schema)
    va_df = _enforce_types(va_df, feats, schema)
    tr_df = tr_df.sort_values(C.UID)
    va_df = va_df.sort_values(C.UID)
    cat_idx = _cat_idx(feats, schema)

    model = CatBoostRanker(**CFG.CB)
    model.fit(
        Pool(tr_df[feats], tr_df[C.REL], group_id=tr_df[C.UID], cat_features=cat_idx),
        eval_set=Pool(va_df[feats], va_df[C.REL], group_id=va_df[C.UID], cat_features=cat_idx),
        early_stopping_rounds=CFG.STOP,
        use_best_model=True,
    )

    model.save_model(str(CFG.MDL / C.MODEL))
    (CFG.MDL / "feats.json").write_text(json.dumps(feats))
    _save_schema(schema)


def predict():
    cand = read(CFG.RAW / C.CAND)
    pairs = expand(cand)

    base = pd.read_parquet(CFG.PROC / C.PROC, engine='pyarrow')
    train_df = base[base[C.SRC] == C.VAL_T]

    _, _, _, bg, desc = load()
    users = read(CFG.RAW / C.USERS)
    books = read(CFG.RAW / C.BOOKS).drop_duplicates(C.BID)

    df = pairs.merge(users, on=C.UID, how="left").merge(books, on=C.BID, how="left")
    df = add_genre(df, bg)
    df = add_tfidf_infer(df, desc)
    df = add_agg(df, train_df)
    df = fill(df, train_df)

    feats = json.loads((CFG.MDL / 'feats.json').read_text())
    schema = _load_schema()

    for f in feats:
        if f not in df.columns:
            df[f] = C.MISS if schema.get(f) == 'cat' else 0.0

    df = _enforce_types(df, feats, schema)

    df = df.sort_values(C.UID)
    cat_idx = _cat_idx(feats, schema)

    model = CatBoostRanker()
    model.load_model(str(CFG.MDL / C.MODEL))

    preds = model.predict(Pool(df[feats], group_id=df[C.UID], cat_features=cat_idx))
    df['s'] = preds

    res = []
    for uid, g in df.groupby(C.UID):
        top = g.sort_values('s', ascending=False).head(C.K)[C.BID].astype(str).tolist()
        res.append({C.UID: uid, C.LIST: ','.join(top)})

    sub = pd.DataFrame(res)
    sub.to_csv(CFG.SUB / C.SUB, index=False)
    sub.to_csv(C.SUB, index=False)


def main():
    if not (CFG.PROC / C.PROC).exists():
        prepare()
    train()
    predict()

if __name__ == '__main__':
    main()

