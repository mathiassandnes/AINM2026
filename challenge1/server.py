"""
Challenge 1: Tripletex AI Accounting Agent
FastAPI server exposing POST /solve endpoint.
"""

import base64
import hashlib
import json
import logging
import os
import sys
import time
import traceback

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from agent import TripletexAgent

# ── Structured JSON logging for Cloud Logging ────────────────
# Cloud Run parses JSON lines from stderr as structured log entries,
# making them queryable by severity, task_type, task_id, etc.

class StructuredFormatter(logging.Formatter):
    """Emit JSON lines that Cloud Logging parses into structured entries."""
    def format(self, record):
        entry = {
            "severity": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        # Forward any extra fields set via log.info("...", extra={...})
        for key in ("task_id", "task_type", "prompt_text", "tool_name",
                     "tool_args", "tool_result", "tool_ok", "turn",
                     "api_calls", "api_errors", "tool_count", "tool_sequence",
                     "duration_s", "files", "llm_text"):
            if hasattr(record, key):
                entry[key] = getattr(record, key)
        return json.dumps(entry, default=str, ensure_ascii=False)

handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(StructuredFormatter())
logging.root.handlers = [handler]
logging.root.setLevel(logging.INFO)

# Suppress noisy libs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

log = logging.getLogger("server")


# ── Task type classifier ─────────────────────────────────────

TASK_PATTERNS = [
    # Tier 3 FIRST — more specific patterns take priority over generic ones
    ("ledger_analysis",     ["analyser hovudboka", "analyser hovedboken", "analysieren sie das hauptbuch", "analysez le grand livre", "analyze the general ledger", "analyze the ledger",
                             "kostnadskontoane", "kostnadskontoene", "aufwandskonten", "expense accounts", "comptes de charges", "totalkostnadene", "gesamtkosten", "total costs",
                             "analise o livro razão", "analise o livro", "custos totais", "contas de despesa", "maior aumento",
                             "analice el libro mayor", "cuentas de gasto", "mayor aumento"]),
    ("ledger_error",        ["feil i hovudboka", "feil i hovedboken", "errors in the general ledger", "errors in the ledger",
                             "erreurs dans le grand livre", "fehler im hauptbuch", "errores en el libro mayor", "erros no livro razão",
                             "discovered errors", "découvert des erreurs", "descubierto errores", "descobrimos erros", "entdeckt fehler"]),
    ("employee_contract",   ["contrato de trabajo", "arbeidskontrakt", "employment contract", "contrat de travail", "arbeitsvertrag",
                             "tilbudsbrev", "tilbodsbrev", "carta de oferta", "offer letter", "lettre d'offre", "angebotsschreiben",
                             "komplett onboarding", "complete a integracao", "complète l'intégration",
                             "stillingsprosent", "percentagem", "porcentaje", "pourcentage",
                             "arslonn", "årslønn", "salario anual", "annual salary", "salaire annuel"]),
    ("period_closing",      ["månedsavslutning", "monatsabschluss", "month-end closing", "clôture mensuelle", "cierre mensual", "encerramento mensal",
                             "jahresabschluss", "year-end closing", "clôture annuelle", "cierre anual", "encerramento anual"]),
    ("accounting_dimension", ["regnskapsdimensjon", "rekneskapsdimensjon", "dimension comptable", "accounting dimension", "buchhaltungsdimension", "dimensión contable", "dimensão contab", "kostsenter", "marked",
                              "benutzerdefinierte", "custom dimension", "dimensão personalizada", "dimensión personalizada"]),
    ("voucher_posting",     ["bokfør", "bokfor", "buchen sie", "comptabiliser", "purregebyr", "reminder fee", "mahngebühr", "frais de rappel",
                             "cargo por recordatorio", "taxa de lembrete", "forfalt faktura",
                             "journal voucher", "bilagspostering", "écriture comptable",
                             "avskrivning", "abschreibung", "depreciation", "amortissement", "depreciação",
                             "recibo", "kvittering", "receipt", "reçu", "quittung", "despesa de"]),
    ("project_lifecycle",   ["complete project lifecycle", "project lifecycle", "cycle de vie complet", "ciclo de vida completo",
                             "vollständigen projektlebenszyklus", "komplett prosjektlivssyklus", "prosjektsyklus", "prosjektlivssyklus"]),
    # Tier 1 (simple CRUD)
    ("create_employee",     ["nouvel employé", "neuen mitarbeiter", "nuevo empleado", "novo empregado", "ny ansatt", "nytilsett", "ny tilsett", "new employee"]),
    ("create_product",      ["opprett produktet", "créez le produit", "erstellen sie das produkt", "crea el producto", "crie o produto", "create the product", "med produktnummer"]),
    ("create_customer",     ["opprett kunden", "erstellen sie den kunden", "créez le client", "crea el cliente", "crie o cliente", "create the customer"]),
    ("create_supplier",     ["registrer leverandøren", "registrieren sie den lieferanten", "enregistrez le fournisseur", "registre el proveedor", "registe o fornecedor", "register the supplier"]),
    ("create_departments",  ["tre avdelinger", "tre avdelingar", "drei abteilungen", "trois départements", "tres departamentos", "three departments"]),
    ("simple_invoice",      ["opprett og send ein faktura", "opprett og send en faktura", "créez et envoyez", "erstellen und senden", "create and send",
                             "crie e envie uma fatura", "crea y envía una factura"]),
    ("register_payment",    ["registrer full betaling", "enregistrez le paiement", "registrieren sie die zahlung", "register full payment", "registre el pago", "registre o pagamento"]),
    ("credit_note",         ["kreditnota", "kreditnote", "credit note", "note de crédit", "nota de crédito", "avoir", "reklamiert", "reclamou"]),
    # Tier 2 (multi-step)
    ("invoice_3_lines",     ["tre produktlinjer", "drei produktzeilen", "trois lignes", "tres líneas", "three product lines", "três linhas"]),
    ("order_invoice_payment", ["konverter ordren til faktura og registrer", "convertissez la commande en facture et enregistrez", "convert the order to invoice and register"]),
    ("payroll",             ["køyr løn", "kjør lønn", "gehaltsabrechnung", "run payroll", "paie", "nómina", "folha de pagamento"]),
    ("log_hours_invoice",   ["stundensatz", "hourly rate", "timesats", "taux horaire", "tarifa por hora", "taxa horária",
                             "timar for", "timer for", "hours for", "heures pour", "horas para",
                             "log time", "enregistrez le temps", "registre el tiempo", "registre as horas",
                             "prosjektfaktura", "project invoice"]),
    ("travel_expense",      ["reise", "note de frais", "reisekostn", "travel expense", "gastos de viaje", "despesas de viagem", "conférence", "konferanse"]),
    ("incoming_invoice",    ["mottatt faktura", "leverandørfaktura", "leverandorfaktura", "facture fournisseur", "supplier invoice", "incoming invoice",
                             "factura.*proveedor", "fatura do fornecedor", "reçu la facture", "recebemos a fatura",
                             "fatura inv-", "faktura inv-", "invoice inv-", "facture inv-", "lieferantenrechnung",
                             "leverandorfaktura", "fatura de fornecedor"]),
    ("reverse_payment",     ["returnert av banken", "devuelto por el banco", "retourné par la banque", "returned by the bank", "reverser betaling", "devolvido pelo banco"]),
    ("project_fixed_price", ["preço fixo", "prix fixe", "festpris", "fastpris", "fixed price", "precio fijo", "milestone"]),
    ("project_create",      ["opprett prosjekt", "erstellen sie das projekt", "créez le projet", "create the project", "crea el proyecto", "crie o projeto"]),
    ("bank_reconciliation", ["avstem bankutskrift", "bankutskrift", "bank reconcil", "concilia el extracto", "concilie o extrato",
                             "rapprochez le relevé", "bankabstimmung", "bank statement"]),
]


def classify_task(prompt: str) -> str:
    """Classify task type from prompt keywords. Returns best match or 'unknown'."""
    lower = prompt.lower()
    for task_type, keywords in TASK_PATTERNS:
        for kw in keywords:
            if kw in lower:
                return task_type
    return "unknown"


def task_id_from_prompt(prompt: str) -> str:
    """Generate a stable short ID from prompt to track same task across submissions."""
    # Use first 100 chars + key numbers to fingerprint (ignores name/org variations)
    return hashlib.md5(prompt.encode()).hexdigest()[:8]


app = FastAPI(title="NM i AI - Tripletex Agent")


@app.post("/solve")
async def solve(request: Request):
    t0 = time.time()
    body = await request.json()
    prompt = body["prompt"]
    files = body.get("files", [])
    creds = body["tripletex_credentials"]
    base_url = creds["base_url"]
    token = creds["session_token"]

    task_type = classify_task(prompt)
    tid = task_id_from_prompt(prompt)

    # Decode any attached files
    decoded_files = []
    for f in files:
        decoded_files.append({
            "filename": f["filename"],
            "data": base64.b64decode(f["content_base64"]),
            "mime_type": f["mime_type"],
        })

    file_info = [(f["filename"], f["mime_type"], len(f["data"])) for f in decoded_files]
    log.info(f"[{tid}] TASK START | type={task_type}",
             extra={"task_id": tid, "task_type": task_type,
                     "prompt_text": prompt, "files": file_info})

    try:
        agent = TripletexAgent(base_url=base_url, session_token=token)
        await agent.solve(prompt, decoded_files, task_id=tid, task_type=task_type)
        duration = time.time() - t0
        log.info(f"[{tid}] TASK END | type={task_type} calls={agent.client.call_count} "
                 f"errors={agent.client.error_count} duration={duration:.1f}s",
                 extra={"task_id": tid, "task_type": task_type,
                        "api_calls": agent.client.call_count,
                        "api_errors": agent.client.error_count,
                        "duration_s": round(duration, 1)})
    except Exception:
        duration = time.time() - t0
        log.error(f"[{tid}] AGENT EXCEPTION | type={task_type} duration={duration:.1f}s\n{traceback.format_exc()}",
                  extra={"task_id": tid, "task_type": task_type, "duration_s": round(duration, 1)})

    sys.stderr.flush()
    return JSONResponse({"status": "completed"})


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
