from __future__ import annotations

from typing import Any

import gradio as gr

from utils.demo_logic import run_demo_logic


def create_demo() -> gr.Blocks:
    def analyze_claim(
        claim_id: str,
        description: str,
        amount: float,
        policy_limit: float,
        image: Any,
    ) -> tuple[str, str, str, str]:
        try:
            result = run_demo_logic(claim_id, description, amount, policy_limit, image)

            decision = str(result.get("decision", ""))
            risk = str(result.get("risk_level", ""))
            score = str(result.get("fraud_score", ""))
            explanation = str(result.get("explanation", ""))

            assert isinstance(decision, str)
            assert isinstance(risk, str)
            assert isinstance(score, str)
            assert isinstance(explanation, str)

            return decision, risk, score, explanation
        except Exception as e:
            return "ERROR", "UNKNOWN", "0.0", f"{type(e).__name__}: {e}"

    with gr.Blocks(
        analytics_enabled=False,
        title="AI Insurance Claim Decision Demo",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("## AI Insurance Claim Decision Demo (Standalone)")

        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("### Claim form")
                claim_id = gr.Textbox(label="Claim ID", value="HF-DEMO-001")
                description = gr.Textbox(
                    label="Description",
                    placeholder="Describe the incident, damage, and context.",
                    lines=6,
                    container=True,
                )

                image_input = gr.Image(
                    label="📷 Upload Claim Image (optional)",
                    type="numpy",
                    height=180,
                )

                amount = gr.Number(
                    label="Claim amount (USD)",
                    value=250.0,
                    minimum=0,
                    precision=2,
                    container=True,
                )
                policy_limit = gr.Number(
                    label="Policy limit (USD)",
                    value=1000.0,
                    minimum=0,
                    precision=2,
                    container=True,
                )

                analyze_btn = gr.Button("🚀 Analyze Claim", variant="primary")

                with gr.Row():
                    gr.Button("⚡ Normal claim").click(
                        fn=lambda: ("HF-001", "Minor screen crack", 200, 1000, None),
                        inputs=[],
                        outputs=[claim_id, description, amount, policy_limit, image_input],
                    )
                    gr.Button("🚨 Fraud attempt").click(
                        fn=lambda: ("HF-002", "No damage but high claim", 900, 1000, None),
                        inputs=[],
                        outputs=[claim_id, description, amount, policy_limit, image_input],
                    )
                    gr.Button("🔥 Major damage").click(
                        fn=lambda: ("HF-003", "Mobile screen major crack", 800, 1000, None),
                        inputs=[],
                        outputs=[claim_id, description, amount, policy_limit, image_input],
                    )

                input_summary = gr.Markdown()
                status = gr.Markdown("Ready.")

            with gr.Column(scale=4):
                gr.Markdown("### Results")

                decision_md = gr.Markdown()
                risk_md = gr.Markdown()
                score_md = gr.Markdown()
                explanation_md = gr.Markdown()

        analyze_btn.click(
            fn=analyze_claim,
            inputs=[claim_id, description, amount, policy_limit, image_input],
            outputs=[decision_md, risk_md, score_md, explanation_md],
        )

        demo.queue()

    return demo
