from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from app.agents.state import AgentState
import os
import json
import logging
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.1, max_retries=3)

def reporter_node(state: AgentState):
    """
    Reads strictly VALIDATED claims (status="supported") and pre-computed structured evidence to synthesize
    an executive-ready Markdown EDA report. The LLM acts as an Evidence Interpreter ONLY.
    """
    run_id = state.get("run_id", "default")
    workspace_dir = state.get("workspace_dir")
    
    validated_claims = state.get("validated_claims", [])
    evidence = state.get("eda_evidence", {})
    
    # Format validated claims into bullet points
    claims_text = "\n".join([f"- [{c['category'].upper()}] {c['claim']}" for c in validated_claims])
    if not claims_text:
        claims_text = "- No validated claims passed verification."

    evidence_summary = json.dumps(evidence, indent=2, default=str)
    
    chart_paths = state.get("chart_paths", [])
    if workspace_dir:
        plots_dir = os.path.join(workspace_dir, "plots")
        report_file_path = os.path.join(workspace_dir, "reports", "final_report.md")
    else:
        plots_dir = os.path.join("sandbox", "plots")
        report_file_path = "reports/final_report.md"
        
    if os.path.exists(plots_dir):
        file_charts = [os.path.join(plots_dir, f).replace('\\', '/') for f in os.listdir(plots_dir) if f.endswith(".html")]
        chart_paths = list(set(chart_paths + file_charts))
        
    system_prompt = """
    You are an Expert Data Science Auditor and Executive Technical Reporter.
    Your objective is to write an executive-ready Markdown Exploratory Data Analysis (EDA) report based STRICTLY on the supplied pre-computed evidence and validated claims.

    STRICT EVIDENCE-ONLY RULES:
    1. NEVER calculate or invent any numerical statistics. Use ONLY numbers from the provided validated claims and evidence object.
    2. NEVER make unsupported claims. Include ONLY claims provided in the VALIDATED CLAIMS section.
    3. If evidence is insufficient for any query or claim, state: "Insufficient evidence to determine."
    4. DO NOT infer causation from correlation.
    5. DO NOT claim statistical significance without a statistical test showing p_adj < 0.05.
    6. DO NOT use raw LaTeX syntax (`$...$`, `\\text{{...}}`, etc.). Use clean GitHub Flavored Markdown (GFM).

    VALIDATED CLAIMS (STRICTLY SUPPORTED):
    {claims_text}

    STRUCTURED EDA EVIDENCE (GROUND TRUTH):
    {evidence_summary}

    REPORT STRUCTURE:
    1. **Executive Summary & Headline Data Quality Score**
    2. **Data Profiling & Quality Discoveries (Missingness, Duplicates, PII Shield)**
    3. **Distribution & Shape Analysis (Quantiles, IQR, Skewness, Normality, Outliers)**
    4. **Relationship & Bivariate Findings (Correlations, Group Tests, Simpson's Paradox)**
    5. **Machine Learning Readiness & Anomaly Audit (Imbalance, Multicollinearity, Leakage)**
    6. **Actionable Recommendations & Next Steps**
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Synthesize the validated claims and structured evidence into the final Markdown report.")
    ])
    
    chain = prompt | llm
    
    try:
        response = chain.invoke({
            "claims_text": claims_text,
            "evidence_summary": evidence_summary
        })
        report_content = response.content if isinstance(response.content, str) else response.content[0].get("text", str(response.content))
    except Exception as e:
        logging.warning(f"[run_id={run_id}] LLM report synthesis fallback: {e}")
        report_content = f"# Executive Exploratory Data Analysis Report\n\n*(LLM synthesis offline: {e})*\n\n## Verified Claims\n\n{claims_text}"

    if chart_paths:
        report_content += "\n\n## Interactive Visualizations\n\n"
        for path in sorted(set(chart_paths)):
            filename = os.path.basename(path)
            if workspace_dir:
                chart_src = f"/api/plots/{run_id}/plots/{filename}"
            elif "sandbox" in path:
                chart_src = f"/api/sandbox/plots/{filename}"
            else:
                chart_src = f"/api/plots/{filename}"
            report_content += f'<iframe src="{chart_src}" width="100%" height="600" style="border:none; margin-bottom: 20px;"></iframe>\n\n'
            
    os.makedirs(os.path.dirname(report_file_path), exist_ok=True)
    with open(report_file_path, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    return {
        "messages": [AIMessage(content=f"Final report successfully generated in {report_file_path}")]
    }
