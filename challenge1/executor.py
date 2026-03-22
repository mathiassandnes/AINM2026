"""
Deterministic step executor for the two-phase planner.

Takes Phase 1 (operations + find_match) + Phase 2 (filled params) and
executes API calls sequentially with variable substitution.

For complex operations (vouchers, incoming invoices, etc.), delegates to
the composite tools in tools.py rather than reimplementing API logic.
"""

import json
import logging
import re

from tripletex import TripletexClient
from api_schemas import OPERATIONS

log = logging.getLogger(__name__)


# ── Variable Resolution ──────────────────────────────────────────

def _resolve_path(data, path_parts: list[str]):
    """Navigate nested dict/list: ['values', '[0]', 'id'] → data.values[0].id"""
    current = data
    for part in path_parts:
        if current is None:
            return None
        m = re.match(r"\[(\d+)\]", part)
        if m:
            idx = int(m.group(1))
            if isinstance(current, list) and idx < len(current):
                current = current[idx]
            else:
                return None
        elif isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def _parse_path(path_str: str) -> list[str]:
    """Parse 'values[0].id' → ['values', '[0]', 'id']."""
    parts = []
    for segment in path_str.split("."):
        m = re.match(r"([a-zA-Z_]\w*)((?:\[\d+\])*)", segment)
        if m:
            if m.group(1):
                parts.append(m.group(1))
            for idx_match in re.finditer(r"\[\d+\]", m.group(2)):
                parts.append(idx_match.group())
        else:
            parts.append(segment)
    return parts


def resolve_ref(ref_str: str, step_results: dict, today: str):
    """Resolve a $step_N.path reference to its value."""
    if ref_str == "$TODAY":
        return today

    m = re.match(r"\$step_(\d+)\.(.*)", ref_str)
    if not m:
        return ref_str  # Not a reference

    step_idx = int(m.group(1))
    path_str = m.group(2)
    key = f"step_{step_idx}"

    result = step_results.get(key)
    if result is None:
        log.warning(f"Ref {ref_str}: step_{step_idx} not found in results")
        return None

    path_parts = _parse_path(path_str)
    resolved = _resolve_path(result, path_parts)
    if resolved is None:
        log.warning(f"Ref {ref_str}: could not resolve path '{path_str}' in result")
    return resolved


def _substitute(obj, step_results: dict, today: str):
    """Recursively substitute $step_N references in a data structure."""
    if isinstance(obj, str):
        # Strip <UNKNOWN> placeholders from Phase 2
        if obj == "<UNKNOWN>":
            return None
        # Handle LLM-invented "||" fallback syntax: "$step_0.values[0].id || $step_1.value.id"
        if "||" in obj and "$step_" in obj:
            obj = obj.split("||")[0].strip()
        if obj.startswith("$"):
            return resolve_ref(obj, step_results, today)
        # Embedded refs in strings (e.g. path templates)
        def replace_match(m):
            val = resolve_ref(m.group(0), step_results, today)
            return str(val) if val is not None else m.group(0)
        if "$" in obj:
            return re.sub(r"\$(?:step_\d+\.[\w\[\]]+|TODAY)", replace_match, obj)
        return obj
    elif isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            substituted = _substitute(v, step_results, today)
            if substituted is None:
                continue  # Drop keys with None/<UNKNOWN> values
            if isinstance(substituted, dict) and not substituted:
                continue  # Drop empty dict refs (e.g. department: {})
            result[k] = substituted
        return result
    elif isinstance(obj, list):
        return [_substitute(item, step_results, today) for item in obj]
    return obj


# ── Find Match (for GET results) ─────────────────────────────────

def _get_nested(item: dict, field: str):
    """Get a possibly nested field value: 'customer.name' -> item['customer']['name']."""
    parts = field.split(".")
    val = item
    for p in parts:
        # Handle array indexing like orderLines[0]
        m = re.match(r"(\w+)\[(\d+)\]", p)
        if m:
            val = val.get(m.group(1), [])
            idx = int(m.group(2))
            val = val[idx] if isinstance(val, list) and idx < len(val) else None
        elif isinstance(val, dict):
            val = val.get(p)
        else:
            return None
        if val is None:
            return None
    return val


def _apply_find_match(response: dict, find_match: dict) -> tuple[dict, str]:
    """Filter a GET response to find a specific item matching criteria.

    Returns (filtered_response, description_string).
    """
    if not find_match:
        return response, ""

    field = find_match.get("field", "")
    contains = find_match.get("contains", "").lower()

    values = response.get("values", [])
    for item in values:
        # Support nested fields like customer.name, orderLines[0].description
        val = _get_nested(item, field) if "." in field or "[" in field else item.get(field)
        if val is not None and contains in str(val).lower():
            response = dict(response)
            response["values"] = [item]
            desc = f"matched {field}='{val}' for '{contains}'"
            return response, desc

    # No match found — if there's exactly 1 result, use it anyway (common for invoice lookups by customerId)
    if len(values) == 1:
        response = dict(response)
        desc = f"NO MATCH for {field} contains '{contains}' in 1 results — using only result"
        return response, desc

    desc = f"NO MATCH for {field} contains '{contains}' in {len(values)} results"
    return response, desc


def _infer_find_match(op_id: str, params: dict, description: str) -> dict | None:
    """Auto-infer find_match from operation type and GET params.

    Instead of relying on the planner to specify find_match (error-prone),
    we infer it from the operation and the params that were used to filter.
    """
    # Operations where we filter by name param → match on name field
    name_ops = {"find_customer", "find_supplier", "find_product", "find_project",
                "get_department", "find_department", "find_activity"}
    if op_id in name_ops and params.get("name"):
        return {"field": "name", "contains": params["name"]}

    # find_product by number
    if op_id == "find_product" and params.get("number"):
        return {"field": "number", "contains": params["number"]}

    # find_customer/supplier by orgNumber
    if op_id in ("find_customer", "find_supplier") and params.get("organizationNumber"):
        return {"field": "organizationNumber", "contains": params["organizationNumber"]}

    # get_account by number — match on number
    if op_id == "get_account" and params.get("number"):
        return {"field": "number", "contains": params["number"]}

    # For find_invoice, find_voucher — no find_match, just use first result
    # For get_payment_types, get_cost_categories, etc. — use first result
    return None


# ── Helpers for delegation ────────────────────────────────────────

_account_number_cache: dict[int, int] = {}

async def _get_account_number(client, account_id: int) -> int | None:
    """Look up account number from account ID."""
    if account_id in _account_number_cache:
        return _account_number_cache[account_id]
    result = await client.get(f"/ledger/account/{account_id}", params={"fields": "number"})
    if not result.get("error") and result.get("value", {}).get("number"):
        num = result["value"]["number"]
        _account_number_cache[account_id] = num
        return num
    return None


def _find_entity_id_in_results(step_results: dict | None, entity_type: str) -> int | None:
    """Search step_results for a customer/supplier ID from earlier steps."""
    if not step_results:
        return None
    for key in sorted(step_results.keys()):
        result = step_results[key]
        if not isinstance(result, dict):
            continue
        # POST result: {value: {id: N, ...}}
        val = result.get("value", {})
        if val.get("id"):
            if entity_type == "customer" and (val.get("customerNumber") or val.get("isCustomer")):
                return val["id"]
            if entity_type == "supplier" and (val.get("supplierNumber") or val.get("isSupplier")):
                return val["id"]
            # Invoice result has nested customer/supplier
            if entity_type == "customer" and val.get("customer", {}).get("id"):
                return val["customer"]["id"]
        # GET result: {values: [{id: N, ...}]}
        vals = result.get("values", [])
        if vals:
            first = vals[0]
            if entity_type == "customer":
                if first.get("customerNumber") or first.get("isCustomer"):
                    return first.get("id")
                # Invoice search result has nested customer
                if first.get("customer", {}).get("id"):
                    return first["customer"]["id"]
            if entity_type == "supplier":
                if first.get("supplierNumber") or first.get("isSupplier"):
                    return first.get("id")
    return None


# ── Composite tool delegation ────────────────────────────────────
# Operations that have tricky API requirements are delegated to the
# composite tools in tools.py instead of calling raw API endpoints.

DELEGATED_OPS = {
    "create_voucher",
    "create_incoming_invoice",
    "create_invoice_direct",
    "create_employee",
    "create_employment",
    "create_employment_details",
    "register_payment",
    "create_travel_expense",
}


async def _delegate_to_tool(op_id: str, body: dict, tools, step_results: dict = None) -> dict:
    """Execute an operation via its composite tool instead of raw API."""
    if op_id == "create_voucher":
        # Convert Phase 2 schema format to composite tool format
        postings = []
        for p in body.get("postings", []):
            posting = {}
            # Account: can be {id: N} ref or raw id, with accountNumber as fallback
            acct = p.get("account", {})
            if isinstance(acct, dict) and acct.get("id"):
                posting["account_id"] = acct["id"]
            elif p.get("accountNumber"):
                # Fallback: use account number directly (create_voucher tool resolves number→id and auto-creates)
                posting["account_id"] = p["accountNumber"]
            elif isinstance(acct, dict) and not acct.get("id"):
                # Account ref resolved to None and no accountNumber fallback
                return {"error": True, "message": f"Account reference resolved to empty — account may not exist in this sandbox"}
            amount = p.get("amountGrossCurrency", 0)
            if amount >= 0:
                posting["debit"] = abs(amount)
            else:
                posting["credit"] = abs(amount)
            if p.get("description"):
                posting["description"] = p["description"]
            # Pass through customer_id/supplier_id if present in various formats
            cust_id = None
            if p.get("customer", {}).get("id"):
                cust_id = p["customer"]["id"]
            elif p.get("freeAccountingDimension1", {}).get("id"):
                # Phase 2 sometimes puts customer_id in dimension1 field
                cust_id = p["freeAccountingDimension1"]["id"]
            if cust_id:
                posting["customer_id"] = cust_id

            supp_id = None
            if p.get("supplier", {}).get("id"):
                supp_id = p["supplier"]["id"]
            elif p.get("freeAccountingDimension2", {}).get("id"):
                supp_id = p["freeAccountingDimension2"]["id"]
            if supp_id:
                posting["supplier_id"] = supp_id

            # Auto-inject customer/supplier for accounts that require them
            # Account 1500 (Kundefordringer) needs customer_id
            # Account 2400 (Leverandørgjeld) needs supplier_id
            if not posting.get("customer_id") and posting.get("account_id"):
                acct_num = await _get_account_number(tools.client, posting["account_id"])
                if acct_num == 1500:
                    # Find a customer_id from step_results
                    cid = _find_entity_id_in_results(step_results, "customer")
                    if cid:
                        posting["customer_id"] = cid
                elif acct_num == 2400 and not posting.get("supplier_id"):
                    sid = _find_entity_id_in_results(step_results, "supplier")
                    if sid:
                        posting["supplier_id"] = sid

            # Department
            if p.get("department", {}).get("id"):
                posting["department_id"] = p["department"]["id"]

            # Dimension refs — pass through, but detect department IDs misplaced as dimensions
            for dim in ("freeAccountingDimension1", "freeAccountingDimension2", "freeAccountingDimension3"):
                if p.get(dim, {}).get("id"):
                    dim_num = dim[-1]
                    val = p[dim]["id"]
                    # Skip if already used as customer_id or supplier_id
                    if val == posting.get("customer_id") or val == posting.get("supplier_id"):
                        continue
                    # Check if this is actually a department ID (from get_department step)
                    is_dept = False
                    if step_results:
                        for sk, sr in step_results.items():
                            vals = sr.get("values", [])
                            if vals and vals[0].get("id") == val and "departmentNumber" in vals[0]:
                                is_dept = True
                                break
                            sv = sr.get("value", {})
                            if sv.get("id") == val and "departmentNumber" in sv:
                                is_dept = True
                                break
                    if is_dept and not posting.get("department_id"):
                        posting["department_id"] = val
                    else:
                        posting[f"dimension{dim_num}_id"] = val
            if p.get("project", {}).get("id"):
                posting["project_id"] = p["project"]["id"]
            postings.append(posting)
        return await tools.create_voucher(
            date=body.get("date", ""),
            description=body.get("description", ""),
            postings=postings,
        )

    elif op_id == "create_incoming_invoice":
        # Convert Phase 2 schema to composite tool format
        header = body.get("invoiceHeader", {})
        lines = body.get("orderLines", [])
        account_number = lines[0].get("accountId", 6300) if lines else 6300
        # If accountId is a large number (real ID), we need to reverse-lookup
        # The composite tool accepts account numbers, not IDs
        if isinstance(account_number, int) and account_number > 10000:
            # It's an account ID, not a number — look up the number
            acct = await tools.client.get(f"/ledger/account/{account_number}", params={"fields": "number"})
            if not acct.get("error") and acct.get("value", {}).get("number"):
                account_number = acct["value"]["number"]
        vat_type_id = lines[0].get("vatTypeId", 1) if lines else 1
        return await tools.create_incoming_invoice(
            supplier_id=header.get("vendorId"),
            invoice_date=header.get("invoiceDate", ""),
            due_date=header.get("dueDate", ""),
            amount_incl_vat=header.get("invoiceAmount", 0),
            invoice_number=header.get("invoiceNumber", ""),
            description=header.get("description", ""),
            account_number=account_number,
            vat_type_id=vat_type_id,
        )

    elif op_id == "create_invoice_direct":
        # Delegate to create_invoice_with_lines which handles bank setup
        customer = body.get("customer", {})
        customer_id = customer.get("id") if isinstance(customer, dict) else customer
        orders = body.get("orders", [{}])
        order = orders[0] if orders else {}
        order_lines = []
        for line in order.get("orderLines", []):
            ol = {
                "count": line.get("count", 1),
                "unit_price": line.get("unitPriceExcludingVatCurrency", 0),
                "vat_type_id": line.get("vatType", {}).get("id", 3) if isinstance(line.get("vatType"), dict) else 3,
            }
            # Ensure vat_type_id is int
            try:
                ol["vat_type_id"] = int(ol["vat_type_id"])
            except (ValueError, TypeError):
                ol["vat_type_id"] = 3
            if line.get("description"):
                ol["description"] = line["description"]
            # Product ref: look up by number if it's a string/small number
            prod = line.get("product", {})
            prod_id = prod.get("id") if isinstance(prod, dict) else None
            if prod_id is not None:
                # If it looks like a product number (string or small int), look it up
                try:
                    prod_id_int = int(prod_id)
                    if prod_id_int < 100000:  # Likely a product number, not an ID
                        result = await tools.client.get("/product", params={"number": str(prod_id_int), "fields": "id"})
                        vals = result.get("values", [])
                        if vals:
                            ol["product_id"] = vals[0]["id"]
                    else:
                        ol["product_id"] = prod_id_int
                except (ValueError, TypeError):
                    pass  # Skip invalid product refs
            order_lines.append(ol)
        project_id = order.get("project", {}).get("id") if isinstance(order.get("project"), dict) else None
        return await tools.create_invoice_with_lines(
            customer_id=customer_id,
            invoice_date=body.get("invoiceDate", ""),
            due_date=body.get("invoiceDueDate", ""),
            order_lines=order_lines,
            project_id=project_id,
        )

    elif op_id == "create_employee":
        dept = body.get("department", {})
        dept_id = dept.get("id") if isinstance(dept, dict) else None
        # If dept ref resolved to empty, search step_results for a created department
        if not dept_id and step_results:
            for sk in sorted(step_results.keys(), reverse=True):
                sr = step_results[sk]
                sv = sr.get("value", {})
                if sv.get("id") and "departmentNumber" in sv:
                    dept_id = sv["id"]
                    break
        return await tools.create_employee(
            first_name=body.get("firstName", ""),
            last_name=body.get("lastName", ""),
            email=body.get("email", ""),
            date_of_birth=body.get("dateOfBirth", ""),
            phone_mobile=body.get("phoneNumberMobile", ""),
            employee_number=body.get("employeeNumber", ""),
            department_id=dept_id,
        )

    elif op_id == "create_employment":
        emp = body.get("employee", {})
        emp_id = emp.get("id") if isinstance(emp, dict) else emp
        start_date = body.get("startDate", body.get("start_date", ""))
        if not emp_id:
            return {"error": True, "message": "create_employment: missing employee_id"}
        return await tools.create_employment(
            employee_id=int(emp_id),
            start_date=start_date,
        )

    elif op_id == "create_employment_details":
        emp = body.get("employment", {})
        emp_id = emp.get("id") if isinstance(emp, dict) else emp
        hpd = body.get("hoursPerDay", 0)
        if not hpd:
            hpw = body.get("hoursPerWeek", 0)
            hpd = hpw / 5 if hpw else 0
        return await tools.create_employment_details(
            employment_id=emp_id,
            date=body.get("date", ""),
            percentage=body.get("percentageOfFullTimeEquivalent", 100.0),
            annual_salary=body.get("annualSalary", 0),
            hours_per_day=hpd,
        )

    elif op_id == "register_payment":
        # register_payment is a PUT with path_params and query params
        # The composite tool auto-fetches payment type if not provided
        invoice_id = body.get("invoice_id") or body.get("path_params", {}).get("invoice_id")
        if not invoice_id:
            return {"error": True, "message": "register_payment: missing invoice_id"}
        payment_date = body.get("paymentDate", body.get("payment_date", ""))
        amount = body.get("paidAmount", body.get("amount", 0))
        payment_type_id = body.get("paymentTypeId", body.get("payment_type_id"))
        # Convert string payment_type_id to int if possible
        if isinstance(payment_type_id, str):
            try:
                payment_type_id = int(payment_type_id)
            except (ValueError, TypeError):
                payment_type_id = None
        # Reject obviously invalid payment type IDs (hardcoded small numbers like 1, 2)
        if isinstance(payment_type_id, (int, float)) and payment_type_id < 1000:
            payment_type_id = None  # Let composite tool auto-fetch
        return await tools.register_payment(
            invoice_id=int(invoice_id),
            payment_date=payment_date,
            amount=float(amount),
            payment_type_id=payment_type_id,
        )

    elif op_id == "create_travel_expense":
        emp = body.get("employee", {})
        emp_id = emp.get("id") if isinstance(emp, dict) else emp
        if not emp_id:
            return {"error": True, "message": "create_travel_expense: missing employee_id"}
        costs = []
        for c in body.get("costs", []):
            costs.append({
                "description": c.get("description", ""),
                "amount": c.get("amount", 0),
                "date": c.get("date", ""),
                "category": c.get("category", ""),
            })
        return await tools.create_travel_expense(
            employee_id=int(emp_id),
            title=body.get("title", ""),
            date=body.get("date", ""),
            departure_date=body.get("departure_date", body.get("travelDetails", {}).get("departureDate", "")),
            return_date=body.get("return_date", body.get("travelDetails", {}).get("returnDate", "")),
            per_diem_days=body.get("per_diem_days", 0),
            per_diem_rate=body.get("per_diem_rate", 0),
            costs=costs or None,
        )

    return {"error": True, "message": f"No delegation handler for {op_id}"}


# ── Main Executor ─────────────────────────────────────────────────

async def execute_plan(
    client: TripletexClient,
    phase1_steps: list[dict],
    phase2_params: dict,
    today: str,
    task_id: str = "",
    trace=None,
    tools=None,
) -> dict:
    """Execute the plan: iterate operations, fill params, make API calls.

    For complex operations (vouchers, incoming invoices, etc.), delegates
    to composite tools instead of calling raw API endpoints.

    Args:
        phase1_steps: List of {operation, description, find_match?, path_params?}
        phase2_params: Dict of {step_N: {filled params}} from Phase 2
        today: Today's date string
        task_id: For logging
        trace: Optional TaskTrace for detailed logging
        tools: TripletexTools instance for delegated operations

    Returns:
        {success, completed_steps, total_steps, results, error?}
    """
    step_results = {}
    tag = f"[{task_id}]" if task_id else ""
    total = len(phase1_steps)

    for i, step in enumerate(phase1_steps):
        op_id = step["operation"]
        op = OPERATIONS[op_id]
        method = op["method"]
        path = op["path"]
        description = step.get("description", "")
        find_match = step.get("find_match")
        step_key = f"step_{i}"

        # Get filled params from Phase 2 (if any)
        filled = phase2_params.get(step_key, {})

        # Smart skip: if this is a create step and the preceding find step
        # already found the entity, skip the create and reuse the find result.
        _skip_step = False
        if method == "POST" and i > 0:
            _FIND_CREATE_PAIRS = {
                "create_customer": "find_customer",
                "create_supplier": "find_supplier",
                "create_employee": "find_employee",
                "create_product": "find_product",
                "create_department": "get_department",
                "create_account": "get_account",
            }
            find_op = _FIND_CREATE_PAIRS.get(op_id)
            if find_op:
                # For accounts, only check the immediately preceding step (pattern is always get_account → create_account)
                # For other entities, scan backwards to find the matching find step
                if op_id == "create_account":
                    search_range = [i - 1] if i > 0 else []
                else:
                    search_range = range(i - 1, -1, -1)
                for prev_i in search_range:
                    prev_step = phase1_steps[prev_i]
                    prev_key = f"step_{prev_i}"
                    if prev_step["operation"] == find_op and prev_key in step_results:
                        prev_result = step_results[prev_key]
                        prev_values = prev_result.get("values", [])
                        if prev_values:
                            # Found entity — skip create, reuse as {value: first_match}
                            log.info(f"Step {i} ({op_id}): skipping create — {find_op} already found entity in step {prev_i}")
                            step_results[step_key] = {"value": prev_values[0]}
                            if trace:
                                trace.exec_step(i, total, op_id, method, path,
                                    None, None, step_results[step_key], True,
                                    find_match_result=f"SKIPPED — reusing {find_op} result from step {prev_i}")
                            _skip_step = True
                            break
        if _skip_step:
            continue

        # Resolve path parameters (e.g. /order/{order_id}/:invoice)
        path_params = step.get("path_params") or {}
        # Phase 2 might also have path_params
        if isinstance(filled, dict) and "path_params" in filled:
            path_params.update(filled.pop("path_params"))
        # Substitute refs in path params
        path_params = _substitute(path_params, step_results, today)
        for param_name, param_val in path_params.items():
            if param_val is None:
                error_msg = f"Step {i} ({op_id}): path param '{param_name}' resolved to None"
                if trace:
                    trace.exec_result(False, i, total, error_msg)
                return {"success": False, "completed_steps": i, "total_steps": total,
                        "results": step_results, "error": error_msg}
            path = path.replace(f"{{{param_name}}}", str(param_val))

        # Fail if path still has unresolved {params}
        if "{" in path and "}" in path:
            error_msg = f"Step {i} ({op_id}): unresolved path params in '{path}'"
            if trace:
                trace.exec_result(False, i, total, error_msg)
            return {"success": False, "completed_steps": i, "total_steps": total,
                    "results": step_results, "error": error_msg}

        # Build query params
        params = dict(op.get("default_params", {}))
        if method == "GET" and isinstance(filled, dict):
            params.update(_substitute(filled, step_results, today))
            # Strip None values and <UNKNOWN> placeholders from Phase 2
            params = {k: v for k, v in params.items()
                      if v is not None and v != "<UNKNOWN>"}
            # Always use wide date range for find_invoice — planner often picks too narrow
            if op_id == "find_invoice":
                params["invoiceDateFrom"] = "2024-01-01"
                params["invoiceDateTo"] = today
            if op_id == "find_voucher":
                if not params.get("dateFrom"):
                    params["dateFrom"] = "2024-01-01"
                if not params.get("dateTo"):
                    params["dateTo"] = today
            # Fix exclusive date range: dateTo must be > dateFrom
            if params.get("dateFrom") and params.get("dateTo") and params["dateFrom"] == params["dateTo"]:
                from datetime import date as _dt, timedelta as _td
                try:
                    d = _dt.fromisoformat(params["dateTo"])
                    params["dateTo"] = (d + _td(days=1)).isoformat()
                except ValueError:
                    pass
        elif method == "PUT" and isinstance(filled, dict):
            if "params" in filled:
                params.update(_substitute(filled["params"], step_results, today))
            elif op.get("params_schema") and not op.get("body_schema"):
                # PUT with only params_schema (e.g. create_credit_note) — filled IS the params
                params.update(_substitute(filled, step_results, today))

        # Build body
        body = None
        if method == "POST" and filled:
            body = _substitute(filled, step_results, today)
        elif method == "PUT" and isinstance(filled, dict):
            if "body" in filled:
                body = _substitute(filled["body"], step_results, today)
            elif op.get("body_schema") and not op.get("params_schema"):
                body = _substitute(filled, step_results, today)

        # Non-default params for logging
        extra_params = {k: v for k, v in params.items() if k not in op.get("default_params", {})} if params else None

        # Execute
        try:
            # Delegate complex operations to composite tools
            delegate_data = body if body is not None else (params if method == "PUT" else None)
            if op_id in DELEGATED_OPS and tools is not None and delegate_data is not None:
                # Inject resolved path params into delegate_data for PUT ops
                if method == "PUT" and isinstance(delegate_data, dict) and path_params:
                    delegate_data = dict(delegate_data)
                    for pk, pv in path_params.items():
                        delegate_data[pk] = pv
                response = await _delegate_to_tool(op_id, delegate_data, tools, step_results)
            elif method == "GET":
                response = await client.get(path, params=params or None)
            elif method == "POST":
                # Auto bank setup for order/invoice operations
                if op_id in ("create_order", "create_invoice_direct") and tools:
                    await tools._ensure_bank_account()
                response = await client.post(path, json=body)
            elif method == "PUT":
                response = await client.put(path, json=body, params=params if params != op.get("default_params", {}) else None)
            elif method == "DELETE":
                response = await client.delete(path)
            else:
                response = {"error": f"Unknown method: {method}"}
        except Exception as e:
            import traceback as _tb
            err_detail = f"{type(e).__name__}: {e}\n{_tb.format_exc()}"
            log.error(f"Step {i} ({op_id}) exception: {err_detail}")
            if trace:
                trace.exec_error(i, total, op_id, err_detail[:500])
            return {
                "success": False,
                "completed_steps": i,
                "total_steps": total,
                "results": step_results,
                "error": f"Step {i} ({op_id}): {err_detail}",
            }

        is_error = bool(response.get("error"))

        # Apply find_match for GET results — auto-infer from operation + params
        find_match_desc = ""
        if method == "GET" and not is_error:
            # Use explicit find_match from planner if provided (legacy support)
            inferred_match = find_match
            if not inferred_match:
                # Auto-infer: use the GET param that identifies the entity
                inferred_match = _infer_find_match(op_id, params, description)
            if inferred_match:
                response, find_match_desc = _apply_find_match(response, inferred_match)

        # Trace logging
        if trace:
            trace.exec_step(
                i, total, op_id, method, path,
                extra_params, body, response, not is_error,
                find_match_result=find_match_desc,
            )

        if is_error:
            error_msg = f"Step {i} ({op_id}, {method} {path}): {response.get('message', response)}"
            if trace:
                trace.exec_result(False, i, total, error_msg)
            return {
                "success": False,
                "completed_steps": i,
                "total_steps": total,
                "results": step_results,
                "error": error_msg,
                "error_response": response,
            }

        step_results[step_key] = response

    if trace:
        trace.exec_result(True, total, total)

    return {
        "success": True,
        "completed_steps": total,
        "total_steps": total,
        "results": step_results,
    }
