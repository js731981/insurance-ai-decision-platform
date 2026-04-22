import datetime

import gradio as gr
import numpy as np


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
    similar = retrieve_similar(desc or "")

    rag_text = ""
    approved = 0
    rejected = 0
    for s in similar:
        rag_text += f"- {s['id']} → {s['decision']} (${s['amount']})\n"
        if s.get("decision") == "APPROVED":
            approved += 1
        else:
            rejected += 1

    rag_summary_body = f"""
* Similar claims found: {len(similar)}
* Approved: {approved}
* Rejected: {rejected}
  """
    rag_markdown = rag_summary_body + ("\n" + rag_text if rag_text else "")
    rag_hits = bool(similar)

    result = {
        "decision": decision,
        "fraud_score": fraud_score,
        "severity": severity,
        "cnn_label": cnn_label,
        "rag_summary": rag_markdown.strip() if rag_hits else "",
        "amount": amount_f,
        "policy_limit": policy_f,
        "llm_used": False,
    }

    pipeline = {
        "CNN": "✔ Used" if result.get("cnn_label") else "✖ Not Used",
        "Rules": "✔ Used",
        "RAG": "✔ Used" if result.get("rag_summary") else "✖ Not Used",
        "LLM": "✔ Used" if result.get("llm_used") else "✖ Not Used",
    }

    timeline = [
        "✔ Claim Received",
        "✔ RAG Retrieval" if result.get("rag_summary") else "✖ RAG Skipped",
        "✔ CNN Analysis" if result.get("cnn_label") else "✖ CNN Skipped",
        "✔ Rule Engine",
        "✔ Decision Fusion",
        f"🎯 Final: {result.get('decision')}",
    ]

    policy_lim = result.get("policy_limit") or 0
    amt = result.get("amount") or 0
    if policy_lim and policy_lim > 0:
        ratio_s = f"{round(amt / policy_lim, 2)}"
    else:
        ratio_s = "N/A (no policy limit)"

    fs = result.get("fraud_score")
    fs_s = f"{float(fs):.2f}" if fs is not None else "—"

    risk_md = f"""
* **Fraud Score**: {fs_s}
* **Severity**: {result.get("severity")}
* **Claim Ratio**: {ratio_s}
"""

    timeline_md = "\n".join(f"* {row}" for row in timeline)
    pipeline_md = "\n".join(f"* **{k}**: {v}" for k, v in pipeline.items())

    band = risk_band(fraud_score)
    pipeline_exp = pipeline_explanation(severity, amount_f)
    rules_exp = rule_explanation(severity, amount_f, policy_f)
    confidence_md = confidence_breakdown(cnn_conf if cnn_conf is not None else 0.0, fraud_score)
    cf = counterfactual(amount_f, policy_f)

    badge = decision_badge(decision)
    risk_html = risk_bar_html(fraud_score)
    note = narrative(fraud_score)

    summary = f"""

* Claim ID: {claim_id}
* Decision: {decision}
* Risk Level: {severity}
* Risk Band: {band}
* Fraud Score: {fraud_score}
* Narrative: {note}
  """

    label_disp = label if label is not None else "—"
    conf_disp = cnn_conf if cnn_conf is not None else "—"
    signals = f"""

* Damage Type: {label_disp} ({conf_disp})
* Claim Amount: {amount_f}
* Policy Limit: {policy_f}
* Image: {cnn_note}
  """

    final_reason = f"""
Decision '{decision}' is based on:
* Severity: {severity}
* Fraud score: {fraud_score}
* Historical patterns: {approved} approved vs {rejected} rejected
  """

    HISTORY.append({"id": claim_id, "desc": desc, "amount": amount_f, "decision": decision})
    log_decision(claim_id, decision, fraud_score)

    cnn_out = cnn_note if not has_image else f"{cnn_note} — {label} ({cnn_conf})"

    return (
        badge,
        f"{fraud_score:.2f}",
        severity,
        cnn_out,
        timeline_md,
        pipeline_md,
        risk_md,
        risk_html,
        summary,
        signals,
        rag_markdown,
        final_reason,
        pipeline_exp,
        rules_exp,
        confidence_md,
        cf,
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
            decision = gr.HTML(label="Decision")
            fraud = gr.Textbox(label="Fraud Score")
            severity = gr.Textbox(label="Severity")
            cnn = gr.Textbox(label="CNN Output")
            timeline_output = gr.Markdown(label="🧭 Decision Timeline")
            pipeline_output = gr.Markdown(label="⚙️ AI Pipeline")
            heatmap_output = gr.Markdown(label="🔥 Risk Heatmap")
            risk_bar = gr.HTML(label="Risk Meter")

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
            risk_bar,
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
    demo.launch()
