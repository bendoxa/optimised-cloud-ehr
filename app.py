"""Optimised cloud EHR: research prototype (Hugging Face Space).

Clinician view: NIN token lookup or probabilistic match, encounter history,
new encounter entry, four-disease risk from the CNN-BiLSTM-Attention model with
per-visit attention weights. Administrator view: LSTM forecast replay and a
live PSO allocation compared with the static reference, plus the simulation
summary. All patient data in this app are synthetic.
"""
import json
import os
import time
from datetime import date
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import cloudmodel as cm
import ehr

HERE = Path(__file__).parent
A = HERE / "assets"
LABELS = ["Hypertension", "Diabetes", "Malaria", "Respiratory infection"]
con = ehr.connect()
ehr.seed_synthetic(con)

# ------------------------------------------------------------- disease model
MODEL, THR = None, np.full(4, 0.5)
if (A / "disease_weights.npz").exists():
    import npmodel  # NumPy forward pass of the trained Keras model; no TensorFlow needed at run time
    meta = json.loads((A / "disease_model.json").read_text())
    MODEL = npmodel.Model(A / "disease_weights.npz")
    THR = np.array(meta["thresholds"])

# ------------------------------------------------------------- cloud model
P = cm.Params()
cal = json.loads((A / "calibration.json").read_text())
for k, v in cal["demand_mapping"].items():
    P.alpha[k], P.beta[k] = v["alpha"], v["beta"]
P.phi = cal["phi"]["value"]; P.theta0 = cal["surrogate"]["theta"][0]; P.theta = tuple(cal["surrogate"]["theta"][1:])
SIGMA = json.loads((A / "forecast_meta.json").read_text())["sigma_e"]
FC = pd.read_csv(A / "forecast_sample.csv")
FAC = pd.read_csv(A / "facilities.csv")
SIM = pd.read_csv(A / "sim_summary.csv")
TAU = 0.0019


# ------------------------------------------------------------- clinician actions
def lookup(nin, first, last, dob, sex):
    t0 = time.perf_counter()
    if nin:
        r = ehr.find_by_nin(con, nin)
        how = "NIN token match"
        cands = [r] if r else []
    else:
        cands = ehr.probable_matches(con, first or "", last or "", dob or "", sex or "")
        how = "probabilistic match (name + date of birth, blocked on birth year and sex)"
    ms = (time.perf_counter() - t0) * 1e3
    if not cands:
        return None, f"No patient found ({how}, {ms:.1f} ms).", pd.DataFrame(), None, None
    r = cands[0]
    msg = f"**{r[1].title()} {r[2].title()}**, {r[4].lower()}, born {r[3]} · found by {how} in {ms:.1f} ms · patient key `{r[0][:12]}…`"
    tab, risk, att = history_and_risk(r)
    return r[0], msg, tab, risk, att


def history_and_risk(r):
    encs = ehr.encounters(con, r[0])
    tab = pd.DataFrame(encs, columns=["Date", "Clinic", "SBP", "DBP", "Pulse", "Temp (°C)", "Weight (kg)", "Recorded diagnosis"])
    if MODEL is None or not encs:
        return tab, None, None
    X, M, dates = ehr.build_sequence(r[3], r[4], encs)
    p, alpha = MODEL.predict(X, M)
    alpha = alpha[-len(dates):]
    risk = go.Figure(go.Bar(x=p, y=LABELS, orientation="h", marker_color=["#c0392b" if pi >= t else "#2a78d6" for pi, t in zip(p, THR)],
                            text=[f"{pi:.2f} {'FLAG' if pi >= t else ''}" for pi, t in zip(p, THR)], textposition="outside"))
    for i, t in enumerate(THR):
        risk.add_shape(type="line", x0=t, x1=t, y0=i - 0.4, y1=i + 0.4, line=dict(color="#555", dash="dot"))
    risk.update_layout(title="Disease risk (dotted line = decision threshold)", xaxis_range=[0, 1.1], height=280, margin=dict(l=10, r=10, t=40, b=10))
    att = go.Figure(go.Bar(x=dates, y=alpha, marker_color="#6250d6"))
    att.update_layout(title="Attention weight per visit (which visits drove the prediction)", height=250, margin=dict(l=10, r=10, t=40, b=10))
    return tab, risk, att


def add_encounter(tok, clinic, sbp, dbp, pulse, temp, weight):
    if not tok:
        return "Look up a patient first.", gr.update(), gr.update(), gr.update()
    con.execute("INSERT INTO encounters(token,visit_date,clinic,sbp,dbp,pulse,temp,weight,note) VALUES (?,?,?,?,?,?,?,?,?)",
                (tok, date.today().isoformat(), clinic, sbp or None, dbp or None, pulse or None, temp or None, weight or None, "NEW (no diagnosis yet)"))
    con.commit()
    r = con.execute("SELECT * FROM patients WHERE token=?", (tok,)).fetchone()
    tab, risk, att = history_and_risk(r)
    return "Encounter saved and risk recomputed.", tab, risk, att


def register(nin, first, last, dob, sex, phone):
    if nin and ehr.find_by_nin(con, nin):
        return "A patient with this NIN already exists. No duplicate created."
    dup = ehr.probable_matches(con, first, last, dob, sex)
    if dup:
        return f"Possible duplicate of {dup[0][1].title()} {dup[0][2].title()} (born {dup[0][3]}). Sent to manual review; not created."
    tok = ehr.token(nin) if nin else ehr.token(f"{first}|{last}|{dob}|{time.time()}")
    con.execute("INSERT INTO patients VALUES (?,?,?,?,?,?)", (tok, first.lower(), last.lower(), dob, sex, phone)); con.commit()
    return f"Registered. Patient key `{tok[:12]}…` (the NIN itself is not stored)."


# ------------------------------------------------------------- administrator actions
def replay(fid):
    g = FC[FC.facility_id == fid].iloc[:96 * 4]
    fig = go.Figure()
    fig.add_scatter(y=g.observed, name="Observed requests", line=dict(color="#0b0b0b", width=1.4))
    fig.add_scatter(y=g.forecast, name="LSTM forecast", line=dict(color="#2a78d6", width=1.4))
    fig.add_scatter(y=g.persistence, name="Persistence", line=dict(color="#eda100", width=1, dash="dash"))
    fig.update_layout(title=f"{fid}: next-interval EHR requests, first 4 test days (15-min intervals)", height=330,
                      margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"))
    e = g.observed - g.forecast
    return fig, f"RMSE {np.sqrt((e ** 2).mean()):.1f} requests on this window; σₑ for headroom = {SIGMA:.1f}."


def decide(yhat):
    D, u, n = cm.demand(float(yhat), SIGMA, TAU, P)
    A_ = P.phi * u
    r = cm.pso(D, A_, P, rng=np.random.default_rng(1), trace=True)
    Bt = cm.erlang_b_table(A_, P.x_max[3])
    rows = []
    for name, x in (("Static reference", np.array(P.x0, float)), ("PSO allocation", r["x"])):
        e = cm.evaluate(x, D, Bt, P)
        rows.append([name, int(x[0]), round(x[1], 1), round(x[2], 1), int(x[3]), round(float(e["R"][0]), 1),
                     round(float(e["PB"][0]) * 100, 3), round(float(e["C"][0]), 2), "yes" if e["feasible"][0] else "no"])
    tab = pd.DataFrame(rows, columns=["Allocation", "vCPU", "Memory GB", "Bandwidth Mbps", "Pool", "Response ms", "Blocking %", "Cost/h", "Feasible"])
    conv = go.Figure(go.Scatter(y=r["history"], mode="lines+markers", line=dict(color="#2a78d6")))
    conv.update_layout(title=f"PSO convergence ({r['iters']} iterations)", xaxis_title="Iteration", yaxis_title="Best penalised objective",
                       height=280, margin=dict(l=10, r=10, t=40, b=10))
    return tab, conv


def summary(scen):
    s = SIM[SIM.scenario == scen][["controller", "latency_ms", "SVR_pct", "blocking_pct", "cost", "availability_pct"]]
    return s.round(3).rename(columns={"controller": "Controller", "latency_ms": "Latency ms", "SVR_pct": "SLA violations %",
                                      "blocking_pct": "Blocking %", "cost": "Cost/h", "availability_pct": "Availability %"})


def role_view(request: gr.Request):
    user = (request.username or "admin") if request else "admin"
    is_admin = user.startswith("admin")
    return gr.update(visible=True), gr.update(visible=is_admin), f"Signed in as **{user}** ({'administrator' if is_admin else 'clinician'})"


demo_nins = (A / "demo_nins.txt").read_text().split()[:5] if (A / "demo_nins.txt").exists() else []
with gr.Blocks(title="Optimised Cloud EHR prototype") as demo:
    gr.Markdown("# Optimised Cloud EHR: research prototype\nSynthetic patients only. Built for the PhD study on performance optimisation of cloud-based EHR with deep learning.")
    who = gr.Markdown()
    with gr.Tab("Clinician") as t_clin:
        tok = gr.State()
        with gr.Row():
            nin = gr.Textbox(label="NIN (demo: " + ", ".join(demo_nins[:3]) + ")")
            first = gr.Textbox(label="First name"); last = gr.Textbox(label="Surname")
            dob = gr.Textbox(label="Date of birth (YYYY-MM-DD)"); sex = gr.Dropdown(["FEMALE", "MALE"], label="Sex")
        go_btn = gr.Button("Find patient", variant="primary")
        info = gr.Markdown()
        hist = gr.Dataframe(label="Encounter history", interactive=False)
        with gr.Row():
            risk = gr.Plot(); att = gr.Plot()
        gr.Markdown("### New encounter")
        with gr.Row():
            clinic = gr.Dropdown(ehr.CLINICS, value="GP adult", label="Clinic")
            sbp = gr.Number(label="Systolic", value=120); dbp = gr.Number(label="Diastolic", value=80)
            pulse = gr.Number(label="Pulse (0 = not taken)"); temp = gr.Number(label="Temperature °C (0 = not taken)"); weight = gr.Number(label="Weight kg (0 = not taken)")
        save = gr.Button("Save encounter and re-score"); saved = gr.Markdown()
        with gr.Accordion("Register a new patient", open=False):
            with gr.Row():
                rn = gr.Textbox(label="NIN"); rf = gr.Textbox(label="First name"); rl = gr.Textbox(label="Surname")
                rd = gr.Textbox(label="Date of birth"); rs = gr.Dropdown(["FEMALE", "MALE"], label="Sex"); rp = gr.Textbox(label="Phone")
            rb = gr.Button("Register"); rmsg = gr.Markdown()
        go_btn.click(lookup, [nin, first, last, dob, sex], [tok, info, hist, risk, att])
        save.click(add_encounter, [tok, clinic, sbp, dbp, pulse, temp, weight], [saved, hist, risk, att])
        rb.click(register, [rn, rf, rl, rd, rs, rp], rmsg)
    with gr.Tab("Administrator") as t_admin:
        gr.Markdown("### Workload forecast (LSTM) on held-out data")
        fid = gr.Dropdown(FAC.facility_id.tolist(), value=FAC.facility_id.iloc[0], label="Facility")
        fplot = gr.Plot(); fmsg = gr.Markdown()
        fid.change(replay, fid, [fplot, fmsg])
        gr.Markdown("### Live PSO resource decision")
        yhat = gr.Slider(10, 900, value=600, step=10, label="Forecast requests in next 15 minutes")
        dbtn = gr.Button("Optimise allocation", variant="primary")
        dtab = gr.Dataframe(interactive=False); dconv = gr.Plot()
        dbtn.click(decide, yhat, [dtab, dconv])
        gr.Markdown("### Controller comparison on the test period (100 facilities)")
        scen = gr.Radio(["S0", "S1", "S2", "S3", "SG"], value="S0", label="Scenario")
        stab = gr.Dataframe(interactive=False)
        scen.change(summary, scen, stab)
    demo.load(role_view, None, [t_clin, t_admin, who])
    demo.load(replay, fid, [fplot, fmsg])
    demo.load(summary, scen, stab)

if __name__ == "__main__":
    users = os.environ.get("EHR_USERS", "")  # e.g. "clinician:pass1,admin:pass2" set as a Space secret
    auth = [tuple(u.split(":", 1)) for u in users.split(",") if ":" in u] or None
    demo.launch(auth=auth, server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
