"""EHR store, patient identification and feature building for the prototype.

All patients in this store are synthetic. The patient key is an HMAC-SHA-256
token of the NIN (Eq. 3.70); the NIN itself is never stored.
"""
import hashlib
import hmac
import json
import os
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from rapidfuzz.distance import JaroWinkler

HERE = Path(__file__).parent
DB = Path(os.environ.get("EHR_DB", HERE / "ehr_demo.sqlite"))
KEY = os.environ.get("EHR_HMAC_KEY", "demo-key-change-me").encode()
FEAT = json.loads((HERE / "assets" / "disease_features.json").read_text())
CLINICS = [f.split("=", 1)[1] for f in FEAT["features"] if f.startswith("clinic=")]
T = 8


def token(nin: str) -> str:
    return hmac.new(KEY, nin.strip().encode(), hashlib.sha256).hexdigest()[:32]


def connect():
    con = sqlite3.connect(DB, check_same_thread=False)
    con.execute("""CREATE TABLE IF NOT EXISTS patients(token TEXT PRIMARY KEY, first TEXT, last TEXT, dob TEXT, sex TEXT, phone TEXT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS encounters(id INTEGER PRIMARY KEY AUTOINCREMENT, token TEXT, visit_date TEXT,
                   clinic TEXT, sbp REAL, dbp REAL, pulse REAL, temp REAL, weight REAL, note TEXT)""")
    return con


def seed_synthetic(con, n=300, seed=7):
    """Synthetic demonstration patients with latent conditions that shape their vitals."""
    if con.execute("SELECT COUNT(*) FROM patients").fetchone()[0]:
        return
    rng = np.random.default_rng(seed)
    first = "Osaro Efosa Ivie Uyi Nosa Ehis Itohan Adaeze Chinedu Emeka Ngozi Tunde Ayodele Bisi Segun Aisha Musa Fatima Grace Samuel".split()
    last = "Osagie Igbinedion Omoregie Aigbe Okoro Eze Nwosu Okafor Adeyemi Balogun Bello Yusuf Idahosa Erhabor Ekhator Imasuen".split()
    demo = []
    for i in range(n):
        nin = f"{90000000000 + i:011d}"
        age = int(rng.integers(5, 85))
        dob = date(2024, 1, 1) - timedelta(days=int(age * 365.25 + rng.integers(0, 365)))
        sex = rng.choice(["MALE", "FEMALE"])
        tok = token(nin)
        con.execute("INSERT INTO patients VALUES (?,?,?,?,?,?)", (tok, rng.choice(first), rng.choice(last), dob.isoformat(), sex,
                                                                 "080" + "".join(map(str, rng.integers(0, 10, 8)))))
        htn = rng.random() < (0.05 + 0.006 * max(age - 30, 0))
        dm = rng.random() < (0.02 + 0.003 * max(age - 35, 0))
        d0 = date(2023, 1, 1) + timedelta(days=int(rng.integers(0, 300)))
        for k in range(int(rng.integers(1, 10))):
            mal = rng.random() < 0.18
            rti = rng.random() < 0.10
            clinic = "Cardiology" if htn and rng.random() < 0.4 else "Endocrinology" if dm and rng.random() < 0.4 else \
                "GP children / paediatrics" if age < 16 else rng.choice(["GP adult", "NHIS and staff"])
            sbp = rng.normal(150 if htn else 120, 15); dbp = sbp * 0.62 + rng.normal(0, 6)
            temp = rng.normal(38.2 if mal else 36.6, 0.5) if rng.random() < 0.5 else None
            pulse = rng.normal(96 if (mal or rti) else 78, 10) if rng.random() < 0.6 else None
            wt = rng.normal(78 if dm else 68, 12) if rng.random() < 0.5 else None
            note = ", ".join([n for n, f in (("HTN", htn), ("DM", dm), ("MALARIA", mal), ("URTI", rti)) if f]) or "STABLE"
            con.execute("INSERT INTO encounters(token,visit_date,clinic,sbp,dbp,pulse,temp,weight,note) VALUES (?,?,?,?,?,?,?,?,?)",
                        (tok, (d0 + timedelta(days=30 * k + int(rng.integers(0, 20)))).isoformat(), clinic, round(sbp), round(dbp),
                         None if pulse is None else round(pulse), None if temp is None else round(temp, 1),
                         None if wt is None else round(wt, 1), note))
        demo.append(nin)
    con.commit()
    (HERE / "assets" / "demo_nins.txt").write_text("\n".join(demo[:20]))


def find_by_nin(con, nin):
    return con.execute("SELECT * FROM patients WHERE token=?", (token(nin),)).fetchone()


def probable_matches(con, first, last, dob, sex, limit=5):
    """Fellegi-Sunter style score over name similarity and date of birth, blocked on birth year and sex."""
    rows = con.execute("SELECT * FROM patients WHERE substr(dob,1,4)=? AND sex=?", (dob[:4], sex)).fetchall()
    out = []
    for r in rows:
        jw = JaroWinkler.similarity(f"{first} {last}".lower(), f"{r[1]} {r[2]}".lower())
        s = (10.1 if jw >= 0.92 else -6.6) + (8.4 if r[3] == dob else -3.5)  # log2 weights from the EM fit (Section 4.6)
        out.append((s, r))
    return [r for s, r in sorted(out, key=lambda x: -x[0]) if s > 0][:limit]


def encounters(con, tok):
    return con.execute("SELECT visit_date,clinic,sbp,dbp,pulse,temp,weight,note FROM encounters WHERE token=? ORDER BY visit_date", (tok,)).fetchall()


def build_sequence(dob, sex, encs):
    """Encode the last T encounters exactly as in ubth_prepare.py (27 features, train-set scaler)."""
    names = FEAT["features"]
    sc = FEAT["scaler"]
    rows, prev = [], None
    for i, (vd, clinic, sbp, dbp, pulse, temp, wt, _) in enumerate(encs):
        v = date.fromisoformat(vd)
        f = dict.fromkeys(names, 0.0)
        f["age"] = (v - date.fromisoformat(dob)).days / 365.25
        f["male"] = float(sex == "MALE")
        f["systolic"], f["diastolic"] = sbp, dbp
        f["map"] = (sbp + 2 * dbp) / 3 if sbp and dbp else None
        f["pulse"], f["temperature"], f["weight"] = pulse, temp, wt
        for c, val in (("systolic", sbp), ("pulse", pulse), ("temperature", temp), ("weight", wt)):
            f[f"{c}_missing"] = float(val is None)
        days = (v - prev).days if prev else None
        f["log_days_since_prev"] = float(np.log1p(days or 0)); f["first_visit"] = float(days is None)
        f["log_enc_index"] = float(np.log1p(i))
        f["month_sin"], f["month_cos"] = np.sin(2 * np.pi * v.month / 12), np.cos(2 * np.pi * v.month / 12)
        f["weekday"] = float(v.weekday())
        if f"clinic={clinic}" in f:
            f[f"clinic={clinic}"] = 1.0
        for c in sc["median"]:
            if f[c] is None:
                f[c] = sc["median"][c]
            f[c] = (f[c] - sc["mu"][c]) / sc["sd"][c]
        rows.append([f[n] for n in names]); prev = v
    rows = rows[-T:]
    X = np.zeros((1, T, len(names)), np.float32); M = np.zeros((1, T), np.float32)
    X[0, T - len(rows):] = rows; M[0, T - len(rows):] = 1
    return X, M, [e[0] for e in encs][-T:]
