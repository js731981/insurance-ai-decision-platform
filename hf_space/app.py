import datetime
import os

import gradio as gr
import gradio_client.utils as _gcu
import numpy as np


def _patch_gradio_client_json_schema():
    """HF / Gradio 4.44.x: api_info crashes when schema uses additionalProperties: true (bool)."""
    _orig = _gcu._json_schema_to_python_type

    def _safe(schema, defs=None):
        if schema is True or schema is False:
            return "Any"
        if not isinstance(schema, dict):
            return "Any"
        return _orig(schema, defs)

    _gcu._json_schema_to_python_type = _safe


_patch_gradio_client_json_schema()


HISTORY = [
    {"id": "DEMO-001", "desc": "minor scratch", "amount": 200, "decision": "APPROVED"},
    {"id": "DEMO-002", "desc": "no damage high claim", "amount": 900, "decision": "REJECTED"},
]


AUDIT_LOG = []


def log_decision(claim_id, decision, fraud_score):
    AUDIT_LOG.append(
        {
            "id": claim_id,
            "decision": decision,
            "score": fraud_score,
            "time": str(datetime.datetime.now()),
        }
    )


def cnn_model(desc: str):
    desc_l = (desc or "").lower()
    if "crack" in desc_l:
        return "high_crack", 0.88, "HIGH"
    return "low_damage", 0.65, "LOW"


def rule_engine(amount: float, policy: float, severity: str):
    fraud_score = 0.2

    if severity == "HIGH":
        fraud_score += 0.4

    if policy and amount > 0.8 * policy:
        fraud_score += 0.3

    if fraud_score > 0.7:
        return "INVESTIGATE", fraud_score
    if fraud_score > 0.4:
        return "REVIEW", fraud_score
    return "APPROVED", fraud_score


def retrieve_similar(desc: str):
    results = []
    desc_l = (desc or "").lower()
    words = [w for w in desc_l.split() if w]
    for item in HISTORY:
        item_desc_l = str(item.get("desc", "")).lower()
        if any(word in item_desc_l for word in words):
            results.append(item)
    return results[:3]


def pipeline_status(use_cnn: bool = True, use_rules: bool = True, use_llm: bool = False) -> str:
    return f"""

* CNN: {"✅ Used" if use_cnn else "❌ Skipped"}
* Rules: {"✅ Used" if use_rules else "❌ Skipped"}
* LLM: {"🟡 Simulated" if use_llm else "❌ Not Used"}
  """


def risk_band(score):
    if score < 0.3:
        return "LOW"
    if score < 0.7:
        return "MEDIUM"
    return "HIGH"


def pipeline_explanation(severity, amount):
    return f"""

* CNN analyzed image/description → detected severity '{severity}'
* Rules engine evaluated financial risk using amount ${amount}
* No LLM used (safe deterministic pipeline)
  """


def rule_explanation(severity, amount, policy):
    rules = []

    if severity == "HIGH":
        rules.append("High severity rule triggered")

    if policy and amount > policy * 0.8:
        rules.append("High claim amount rule triggered")

    if not rules:
        rules.append("No strong risk rules triggered")

    return "\n".join([f"- {r}" for r in rules])


def confidence_breakdown(cnn_conf, fraud_score):
    try:
        cnn_conf_f = float(cnn_conf)
    except (TypeError, ValueError):
        cnn_conf_f = 0.0

    try:
        fraud_score_f = float(fraud_score)
    except (TypeError, ValueError):
        fraud_score_f = 0.0

    combined = (cnn_conf_f + fraud_score_f) / 2
    return f"""

* CNN Confidence: {cnn_conf_f:.2f}
* Rule-based Risk Score: {fraud_score_f:.2f}
* Combined Decision Confidence: {combined:.2f}
  """


def counterfactual(amount, policy):
    if policy and amount > policy * 0.8:
        return f"If claim amount was lower (< {policy*0.8:.2f}), risk could reduce"
    return "Claim amount does not significantly affect risk"


def build_explanation(
    claim_id: str,
    desc: str,
    amount: float,
    policy: float,
    label: str,
    severity: str,
    decision: str,
    fraud_score: float,
    similar,
):
    cid = claim_id or "(not provided)"
    explanation = f"""
### 🧠 Decision Summary

* **Claim ID**: {cid}
* **Decision**: {decision}
* **Risk Level**: {severity}
* **Fraud Score**: {fraud_score:.2f}

### 🔍 Key Signals

* **Damage Type (CNN simulation)**: {label}
* **Description**: {desc or "(empty)"}
* **Claim Amount**: {amount:.2f}
* **Policy Limit**: {policy:.2f}

### 📊 Historical Evidence (RAG)
"""
    for s in similar:
        explanation += f"- {s['id']} → {s['decision']} (${s['amount']})\n"

    explanation += f"""

### ⚖️ Final Justification

Decision **{decision}** is based on severity **{severity}**, fraud_score **{fraud_score:.2f}**, and historical claim patterns.
"""
    return explanation


def decision_badge(decision):
    color = {
        "APPROVED": "#4caf50",
        "REVIEW": "#ff9800",
        "INVESTIGATE": "#f44336",
    }.get(decision, "#999")

    return f"<span style='color:white;background:{color};padding:6px 12px;border-radius:10px'>{decision}</span>"


def risk_bar_html(score):
    try:
        score_f = float(score)
    except (TypeError, ValueError):
        score_f = 0.0
    score_f = max(0.0, min(1.0, score_f))

    return f""" <div style="background:#eee;border-radius:10px;padding:4px"> <div style="width:{score_f*100}%;background:#4caf50;height:14px;border-radius:10px"></div> </div>
"""


def narrative(score):
    try:
        score_f = float(score)
    except (TypeError, ValueError):
        score_f = 0.0

    if score_f > 0.7:
        return "High confidence fraud risk detected."
    if score_f > 0.4:
        return "Moderate risk requiring review."
    return "Low risk, likely valid claim."


def _image_uploaded(image) -> bool:
    if image is None:
        return False
    try:
        if isinstance(image, np.ndarray) and image.size == 0:
            return False
    except (TypeError, ValueError):
        return False
    return True


def analyze_claim(claim_id: str, desc: str, amount, policy, image):
    try:
        amount_f = float(amount) if amount is not None else 0.0
    except (TypeError, ValueError):
        amount_f = 0.0

    try:
        policy_f = float(policy) if policy is not None else 0.0
    except (TypeError, ValueError):
        policy_f = 0.0

    llm_used = False
    has_image = _image_uploaded(image)
    if not has_image:
        cnn_label = None
        cnn_note = "📷 No image uploaded"
        label, cnn_conf, severity = None, None, None
        desc_l = (desc or "").lower()
        severity = "HIGH" if "crack" in desc_l else "LOW"
    else:
        cnn_note = "✅ Image processed"
        label, cnn_conf, severity = cnn_model(desc or "")
        cnn_label = label

    decision, fraud_score = rule_engine(amount_f, policy_f, severity)
    fraud_score = float(fraud_score or 0.0)

    similar = retrieve_similar(desc or "")
    rag_data = bool(similar)

    rag_text_lines = []
    approved = 0
    rejected = 0
    for s in similar:
        rag_text_lines.append(f"- {s['id']} → {s['decision']} (${s['amount']})")
        if s.get("decision") == "APPROVED":
            approved += 1
        else:
            rejected += 1
    rag_list_md = "\n".join(rag_text_lines) if rag_text_lines else "_No similar historical claims matched._"

    policy_limit = policy_f if policy_f else 0.0
    if policy_limit > 0:
        claim_ratio = round(amount_f / policy_limit, 2)
    else:
        claim_ratio = "N/A (no policy limit)"

    pipeline_text = f"""
### ⚙️ AI Pipeline

* CNN: {'✔ Used' if cnn_label else '✖ Not Used'}
* Rules: ✔ Used
* RAG: {'✔ Used' if rag_data else '✖ Not Used'}
* LLM: {'✔ Used' if llm_used else '✖ Skipped'}
"""

    timeline_text = f"""
### 🧭 Decision Timeline

* Claim Received
* {'RAG Retrieval' if rag_data else 'RAG Skipped'}
* {'CNN Analysis' if cnn_label else 'CNN Skipped'}
* Rule Engine
* Decision Fusion
* Final: {decision}
"""

    risk_text = f"""
### 🔥 Risk Heatmap

* Fraud Score: {fraud_score:.2f}
* Severity: {severity}
* Claim Ratio: {claim_ratio}
"""

    HISTORY.append({"id": claim_id, "desc": desc, "amount": amount_f, "decision": decision})
    log_decision(claim_id, decision, fraud_score)

    cnn_out = str(cnn_label) if cnn_label else "N/A"
    if has_image and label is not None:
        cnn_out = f"{cnn_note} — {label} ({cnn_conf})"

    severity_s = str(severity) if severity is not None else "N/A"
    decision_s = str(decision) if decision is not None else "N/A"

    band = risk_band(fraud_score)
    note = narrative(fraud_score)
    label_disp = str(label) if label is not None else "N/A (no image — severity inferred from description)"
    try:
        conf_disp = f"{float(cnn_conf):.2f}" if cnn_conf is not None else "—"
    except (TypeError, ValueError):
        conf_disp = "—"

    summary_md = f"""
### 🧠 Decision Summary

* **Claim ID**: {claim_id or "(not provided)"}
* **Decision**: **{decision_s}**
* **Risk level**: {severity_s}
* **Risk band**: {band}
* **Fraud score**: {fraud_score:.2f}
* **Narrative**: {note}
"""

    signals_md = f"""
### 🔍 Key Signals

* **Damage type (CNN simulation)**: {label_disp} (confidence {conf_disp})
* **Description**: {desc or "(empty)"}
* **Claim amount**: ${amount_f:.2f}
* **Policy limit**: ${policy_f:.2f}
* **Image**: {cnn_note}
"""

    rag_md = f"""
### 📊 RAG Summary

* **Similar claims found**: {len(similar)}
* **Approved vs rejected (in sample)**: {approved} approved, {rejected} rejected

{rag_list_md}
"""

    final_reason_md = f"""
### ⚖️ Final Justification

Decision **{decision_s}** reflects severity **{severity_s}**, fraud score **{fraud_score:.2f}**, and historical patterns in this demo store (**{approved}** approved vs **{rejected}** rejected among retrieved neighbors).
"""

    pipeline_exp_md = f"### 🧠 AI Pipeline Explanation\n{pipeline_explanation(severity_s, amount_f).strip()}"
    rules_body = rule_explanation(severity_s, amount_f, policy_f)
    rules_md = f"### ⚖️ Rule Triggers\n\n{rules_body}"
    cnn_conf_for_conf = 0.0 if cnn_conf is None else cnn_conf
    confidence_md = f"### 📊 Confidence Breakdown\n{confidence_breakdown(cnn_conf_for_conf, fraud_score).strip()}"
    cf_md = f"### 🔍 What-if Analysis\n\n{counterfactual(amount_f, policy_f)}"

    return (
        decision_s,
        fraud_score,
        severity_s,
        cnn_out,
        timeline_text.strip(),
        pipeline_text.strip(),
        risk_text.strip(),
        summary_md.strip(),
        signals_md.strip(),
        rag_md.strip(),
        final_reason_md.strip(),
        pipeline_exp_md.strip(),
        rules_md.strip(),
        confidence_md.strip(),
        cf_md.strip(),
    )


with gr.Blocks() as demo:
    gr.Markdown("## 🧠 AI Insurance Claim Decision Demo (MVP+)")

    with gr.Row():
        with gr.Column():
            claim_id = gr.Textbox(label="Claim ID")
            desc = gr.Textbox(label="Description")
            amount = gr.Number(label="Claim Amount")
            policy = gr.Number(label="Policy Limit")
            image = gr.Image(type="numpy", label="Upload Claim Image")

            with gr.Row():
                btn = gr.Button("🚀 Analyze Claim")
                normal_btn = gr.Button("⚡ Normal")
                fraud_btn = gr.Button("🚨 Fraud")

        with gr.Column():
            decision = gr.Textbox(label="Decision")
            fraud = gr.Number(label="Fraud Score")
            severity = gr.Textbox(label="Severity")
            cnn = gr.Textbox(label="CNN / damage signal")
            timeline_output = gr.Markdown(label="🧭 Decision Timeline")
            pipeline_output = gr.Markdown(label="⚙️ AI Pipeline")
            heatmap_output = gr.Markdown(label="🔥 Risk Heatmap")

    with gr.Accordion("🧠 Decision Summary", open=True):
        summary = gr.Markdown()

    with gr.Accordion("🔍 Key Signals"):
        signals = gr.Markdown()

    with gr.Accordion("📊 RAG Summary"):
        rag = gr.Markdown()

    with gr.Accordion("⚖️ Final Justification"):
        final_reason = gr.Markdown()

    with gr.Accordion("🧠 AI Pipeline Explanation"):
        pipeline_exp_out = gr.Markdown()

    with gr.Accordion("⚖️ Rule Triggers"):
        rules_out = gr.Markdown()

    with gr.Accordion("📊 Confidence Breakdown"):
        confidence_out = gr.Markdown()

    with gr.Accordion("🔍 What-if Analysis"):
        cf_out = gr.Markdown()

    def normal_case():
        return "ID-001", "minor scratch", 200, 1000, None

    def fraud_case():
        return "ID-002", "no damage high claim", 900, 1000, None

    normal_btn.click(normal_case, outputs=[claim_id, desc, amount, policy, image])
    fraud_btn.click(fraud_case, outputs=[claim_id, desc, amount, policy, image])

    btn.click(
        analyze_claim,
        inputs=[claim_id, desc, amount, policy, image],
        outputs=[
            decision,
            fraud,
            severity,
            cnn,
            timeline_output,
            pipeline_output,
            heatmap_output,
            summary,
            signals,
            rag,
            final_reason,
            pipeline_exp_out,
            rules_out,
            confidence_out,
            cf_out,
        ],
    )


if __name__ == "__main__":
    _port = int(os.environ.get("PORT") or os.environ.get("GRADIO_SERVER_PORT") or 7860)
    demo.launch(server_name="0.0.0.0", server_port=_port)
