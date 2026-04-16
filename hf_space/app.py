import gradio as gr
import datetime


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


def analyze(claim_id: str, desc: str, amount, policy):
    try:
        amount_f = float(amount) if amount is not None else 0.0
    except (TypeError, ValueError):
        amount_f = 0.0

    try:
        policy_f = float(policy) if policy is not None else 0.0
    except (TypeError, ValueError):
        policy_f = 0.0

    label, cnn_conf, severity = cnn_model(desc or "")
    decision, fraud_score = rule_engine(amount_f, policy_f, severity)
    similar = retrieve_similar(desc or "")
    pipeline = pipeline_status()

    band = risk_band(fraud_score)
    pipeline_exp = pipeline_explanation(severity, amount_f)
    rules_exp = rule_explanation(severity, amount_f, policy_f)
    confidence_md = confidence_breakdown(cnn_conf, fraud_score)
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

    signals = f"""

* Damage Type: {label} ({cnn_conf})
* Claim Amount: {amount_f}
* Policy Limit: {policy_f}
  """

    rag_text = ""
    approved = 0
    rejected = 0
    for s in similar:
        rag_text += f"- {s['id']} → {s['decision']} (${s['amount']})\n"
        if s.get("decision") == "APPROVED":
            approved += 1
        else:
            rejected += 1

    rag_summary = f"""
* Similar claims found: {len(similar)}
* Approved: {approved}
* Rejected: {rejected}
  """

    final_reason = f"""
Decision '{decision}' is based on:
* Severity: {severity}
* Fraud score: {fraud_score}
* Historical patterns: {approved} approved vs {rejected} rejected
  """

    HISTORY.append({"id": claim_id, "desc": desc, "amount": amount_f, "decision": decision})
    log_decision(claim_id, decision, fraud_score)

    return (
        badge,
        f"{fraud_score:.2f}",
        severity,
        f"{label} ({cnn_conf})",
        risk_html,
        summary,
        signals,
        rag_summary + "\n" + rag_text,
        pipeline,
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
            image = gr.Image(label="Upload Claim Image", type="filepath")

            with gr.Row():
                btn = gr.Button("🚀 Analyze Claim")
                normal_btn = gr.Button("⚡ Normal")
                fraud_btn = gr.Button("🚨 Fraud")

        with gr.Column():
            decision = gr.HTML(label="Decision")
            fraud = gr.Textbox(label="Fraud Score")
            severity = gr.Textbox(label="Severity")
            cnn = gr.Textbox(label="CNN Output")
            risk_bar = gr.HTML(label="Risk Meter")

    with gr.Accordion("🧠 Decision Summary", open=True):
        summary = gr.Markdown()

    with gr.Accordion("🔍 Key Signals"):
        signals = gr.Markdown()

    with gr.Accordion("📊 RAG Summary"):
        rag = gr.Markdown()

    with gr.Accordion("⚙️ AI Pipeline"):
        pipeline = gr.Markdown()

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
        return "ID-001", "minor scratch", 200, 1000

    def fraud_case():
        return "ID-002", "no damage high claim", 900, 1000

    normal_btn.click(normal_case, outputs=[claim_id, desc, amount, policy])
    fraud_btn.click(fraud_case, outputs=[claim_id, desc, amount, policy])

    btn.click(
        analyze,
        inputs=[claim_id, desc, amount, policy],
        outputs=[
            decision,
            fraud,
            severity,
            cnn,
            risk_bar,
            summary,
            signals,
            rag,
            pipeline,
            final_reason,
            pipeline_exp_out,
            rules_out,
            confidence_out,
            cf_out,
        ],
    )


if __name__ == "__main__":
    demo.launch()
