"""
Generate structured tool/action schemas from the Tripletex OpenAPI spec.

Each action = one API call with an exact JSON schema the model must follow.
The model picks an action → structured output forces correct fields → we call API.

Key design decisions:
- Strip readonly fields (model can't set them)
- Collapse reference objects to just {"id": int} (that's all the API needs)
- Keep descriptions and enums (model needs them to pick correct values)
- Separate query params from body params
- Output is compact enough for Haiku's context window
"""

import json
import re

SPEC_PATH = "openapi.json"
OUTPUT_PATH = "generated_actions.json"

# ── Endpoint selection ──────────────────────────────────────────────
# (method, path, name, description)

ENDPOINTS = [
    # Employee
    ("GET", "/employee", "search_employees", "Search employees by firstName, lastName, email, employeeNumber"),
    ("POST", "/employee", "create_employee", "Create a new employee"),
    ("GET", "/employee/{id}", "get_employee", "Get employee by ID (use fields=* for all details)"),
    ("PUT", "/employee/{id}", "update_employee", "Update employee (include id+version from GET)"),
    ("POST", "/employee/employment", "create_employment", "Create employment record for employee"),
    ("POST", "/employee/hourlyCostAndRate", "set_hourly_rate", "Set hourly cost/rate for employee"),

    # Customer
    ("GET", "/customer", "search_customers", "Search customers by name, email, organizationNumber"),
    ("POST", "/customer", "create_customer", "Create a new customer"),
    ("GET", "/customer/{id}", "get_customer", "Get customer by ID"),
    ("PUT", "/customer/{id}", "update_customer", "Update customer (include id+version from GET)"),

    # Supplier
    ("GET", "/supplier", "search_suppliers", "Search suppliers by name"),
    ("POST", "/supplier", "create_supplier", "Create a new supplier"),
    ("PUT", "/supplier/{id}", "update_supplier", "Update supplier"),

    # Product
    ("GET", "/product", "search_products", "Search products by name or number"),
    ("POST", "/product", "create_product", "Create a new product"),
    ("PUT", "/product/{id}", "update_product", "Update product"),

    # Department
    ("GET", "/department", "search_departments", "Search departments"),
    ("POST", "/department", "create_department", "Create a new department"),

    # Contact
    ("POST", "/contact", "create_contact", "Create contact person for a customer"),

    # Project
    ("GET", "/project", "search_projects", "Search projects"),
    ("POST", "/project", "create_project", "Create a new project"),
    ("POST", "/project/projectActivity", "link_activity_to_project", "Link activity to project"),

    # Activity
    ("GET", "/activity", "search_activities", "Search activities by name"),
    ("POST", "/activity", "create_activity", "Create a new activity"),

    # Order → Invoice flow
    ("POST", "/order", "create_order", "Create order for a customer"),
    ("POST", "/order/orderline", "add_order_line", "Add line item to an order"),
    ("PUT", "/order/{id}/:invoice", "convert_order_to_invoice", "Convert order to invoice"),
    ("GET", "/invoice", "search_invoices", "Search invoices by date range, customer, etc."),
    ("GET", "/invoice/{id}", "get_invoice", "Get invoice by ID"),
    ("PUT", "/invoice/{id}/:payment", "register_invoice_payment", "Register payment on invoice (uses query params!)"),
    ("PUT", "/invoice/{id}/:createCreditNote", "create_credit_note", "Create credit note for invoice"),
    ("PUT", "/invoice/{id}/:send", "send_invoice", "Send invoice (sendType: EMAIL, EHF, etc.)"),
    ("GET", "/invoice/paymentType", "get_payment_types", "Get invoice payment types"),

    # Incoming (supplier) invoice
    ("POST", "/incomingInvoice", "create_incoming_invoice", "Create incoming/supplier invoice"),

    # Travel expense
    ("POST", "/travelExpense", "create_travel_expense", "Create travel expense (dates go in travelDetails sub-object!)"),
    ("GET", "/travelExpense/{id}", "get_travel_expense", "Get travel expense by ID"),
    ("PUT", "/travelExpense/{id}", "update_travel_expense", "Update travel expense"),
    ("POST", "/travelExpense/cost", "add_travel_cost", "Add cost line to travel expense"),
    ("GET", "/travelExpense/costCategory", "get_cost_categories", "Get travel cost categories"),
    ("GET", "/travelExpense/paymentType", "get_travel_payment_types", "Get travel payment types"),
    ("POST", "/travelExpense/perDiemCompensation", "add_per_diem", "Add per diem compensation (field is 'count' not 'countDays')"),
    ("GET", "/travelExpense/rateCategory", "get_rate_categories", "Get per diem rate categories"),
    ("POST", "/travelExpense/accommodationAllowance", "add_accommodation_allowance", "Add accommodation allowance"),
    ("POST", "/travelExpense/mileageAllowance", "add_mileage_allowance", "Add mileage/km allowance"),
    ("PUT", "/travelExpense/:deliver", "deliver_travel_expense", "Submit/deliver travel expense"),

    # Voucher / Ledger
    ("POST", "/ledger/voucher", "create_voucher", "Create journal voucher with postings"),
    ("GET", "/ledger/voucher", "search_vouchers", "Search vouchers"),
    ("GET", "/ledger/voucher/{id}", "get_voucher", "Get voucher by ID"),
    ("PUT", "/ledger/voucher/{id}/:reverse", "reverse_voucher", "Reverse a voucher (query param: date)"),

    # Accounts & VAT
    ("GET", "/ledger/account", "search_accounts", "Search chart of accounts by number"),
    ("PUT", "/ledger/account/{id}", "update_account", "Update account (e.g. set bankAccountNumber)"),
    ("GET", "/ledger/vatType", "get_vat_types", "Get VAT types"),

    # Salary
    ("POST", "/salary/transaction", "create_salary_transaction", "Create salary/payroll transaction"),
    ("GET", "/salary/type", "get_salary_types", "Get salary types (find Fastlønn=2000, Bonus=2002, etc.)"),

    # Timesheet
    ("POST", "/timesheet/entry", "log_timesheet_hours", "Log hours on timesheet"),
]

# ── Schema names that should collapse to just {id: int} ────────────
# These are reference objects — when creating/updating, you only pass the ID.
REFERENCE_TYPES = {
    "Employee", "Customer", "Supplier", "Product", "Department", "Project",
    "Activity", "Invoice", "Order", "Voucher", "VoucherType", "Document",
    "Currency", "Country", "VatType", "Account", "Payslip", "SalaryType",
    "SalaryTransaction", "TravelExpense", "TravelCostCategory",
    "TravelPaymentType", "TravelExpenseRate", "TravelExpenseRateCategory",
    "Attestation", "AttestationStep", "AccountingDimensionValue",
    "EmployeeCategory", "CloseGroup", "ProductUnit", "Asset",
    "CustomerCategory", "DeliveryAddress",
    "Change", "Link",  # always readonly, but just in case
}

# Fields to always exclude (internal/readonly/noisy)
EXCLUDE_FIELDS = {
    "changes", "url", "displayName", "systemGenerated", "isInactive",
    "createdDate", "lastModifiedDate", "links", "metadata", "htmlUrl",
    "navCode", "type",  # usually readonly enum
}


def resolve_ref(spec: dict, ref: str) -> tuple[str, dict]:
    """Resolve $ref → (schema_name, schema_dict)."""
    parts = ref.lstrip("#/").split("/")
    name = parts[-1]
    result = spec
    for p in parts:
        result = result[p]
    return name, result


def convert_property(spec: dict, field_name: str, field_schema: dict, depth: int = 0) -> dict | None:
    """Convert a single property to a clean JSON schema entry.

    Returns None if the field should be excluded.
    """
    # Resolve $ref
    ref_name = None
    if "$ref" in field_schema:
        ref_name, field_schema = resolve_ref(spec, field_schema["$ref"])

    # Skip readonly
    if field_schema.get("readOnly"):
        return None

    # Skip excluded
    if field_name in EXCLUDE_FIELDS:
        return None

    # Reference types → collapse to {id: int}
    if ref_name and ref_name in REFERENCE_TYPES:
        return {
            "type": "object",
            "properties": {"id": {"type": "integer"}},
            "description": f"{ref_name} reference",
        }

    field_type = field_schema.get("type")

    # Nested object (like TravelDetails, Address, etc.)
    if field_type == "object" or "properties" in field_schema:
        if depth > 2:
            return None
        props = {}
        for sub_name, sub_schema in field_schema.get("properties", {}).items():
            converted = convert_property(spec, sub_name, sub_schema, depth + 1)
            if converted:
                props[sub_name] = converted
        if not props:
            # Could be additionalProperties or empty
            if field_schema.get("additionalProperties"):
                return {"type": "object", "description": field_schema.get("description", "Key-value pairs")}
            return None
        result = {"type": "object", "properties": props}
        if field_schema.get("description"):
            result["description"] = field_schema["description"]
        return result

    # Array
    if field_type == "array":
        items = field_schema.get("items", {})
        if "$ref" in items:
            item_name, items = resolve_ref(spec, items["$ref"])
            # If array of reference types, collapse
            if item_name in REFERENCE_TYPES:
                return {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"id": {"type": "integer"}},
                    },
                    "description": field_schema.get("description", f"Array of {item_name} refs"),
                }
            # If array of data objects (like Posting, Cost, OrderLine), inline
            if items.get("readOnly"):
                return None
            converted_items = convert_schema(spec, items, depth + 1)
            if not converted_items or not converted_items.get("properties"):
                return None
            result = {"type": "array", "items": converted_items}
            if field_schema.get("description"):
                result["description"] = field_schema["description"]
            return result
        else:
            # Primitive array
            return {"type": "array", "items": {"type": items.get("type", "string")}}

    # Primitive
    result = {"type": field_type or "string"}
    if "enum" in field_schema:
        result["enum"] = field_schema["enum"]
    if field_schema.get("description"):
        result["description"] = field_schema["description"]
    if field_schema.get("format"):
        fmt = field_schema["format"]
        if fmt in ("int32", "int64"):
            result["type"] = "integer"
        elif fmt in ("float", "double"):
            result["type"] = "number"
    return result


def convert_schema(spec: dict, schema: dict, depth: int = 0) -> dict | None:
    """Convert an OpenAPI schema to a clean JSON schema for structured output."""
    if schema.get("readOnly"):
        return None

    props = {}
    for field_name, field_schema in schema.get("properties", {}).items():
        converted = convert_property(spec, field_name, field_schema, depth)
        if converted:
            props[field_name] = converted

    if not props:
        return None

    result = {"type": "object", "properties": props}
    if schema.get("description"):
        result["description"] = schema["description"]
    return result


def extract_query_params(operation: dict) -> dict | None:
    """Extract query parameters as a JSON schema."""
    params = operation.get("parameters", [])
    query_params = [p for p in params if p.get("in") == "query"]
    if not query_params:
        return None

    props = {}
    for p in query_params:
        name = p["name"]
        schema = p.get("schema", {"type": "string"})
        entry = {"type": schema.get("type", "string")}
        if p.get("description"):
            entry["description"] = p["description"]
        if "enum" in schema:
            entry["enum"] = schema["enum"]
        props[name] = entry

    return {"type": "object", "properties": props}


def build_action(spec: dict, method: str, path: str, name: str, description: str) -> dict:
    """Build one action schema from the OpenAPI spec."""
    path_item = spec.get("paths", {}).get(path, {})
    operation = path_item.get(method.lower(), {})

    action = {
        "name": name,
        "description": description,
        "method": method,
        "path": path,
    }

    # Path params
    path_params = re.findall(r"\{(\w+)\}", path)
    if path_params:
        action["path_params"] = path_params

    # Query params
    query_schema = extract_query_params(operation)
    if query_schema:
        action["query_params"] = query_schema

    # Body
    request_body = operation.get("requestBody", {})
    if request_body:
        content = request_body.get("content", {})
        for content_type, media in content.items():
            body_ref = media.get("schema", {})
            if "$ref" in body_ref:
                schema_name, schema = resolve_ref(spec, body_ref["$ref"])
            else:
                schema = body_ref
            converted = convert_schema(spec, schema)
            if converted:
                action["body"] = converted
            break

    return action


def main():
    with open(SPEC_PATH) as f:
        spec = json.load(f)

    actions = []
    for method, path, name, description in ENDPOINTS:
        action = build_action(spec, method, path, name, description)
        actions.append(action)

        body_fields = len(action.get("body", {}).get("properties", {}))
        query_fields = len(action.get("query_params", {}).get("properties", {}))
        print(f"  {name:<35} body={body_fields:>2} fields  query={query_fields:>2} params")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(actions, f, indent=2, ensure_ascii=False)

    total_size = len(json.dumps(actions))
    print(f"\n{'='*60}")
    print(f"Generated {len(actions)} actions → {OUTPUT_PATH}")
    print(f"Total size: {total_size:,} bytes ({total_size/1024:.1f} KB)")

    # Show a sample to verify correctness
    print(f"\n{'='*60}")
    print("Sample: create_travel_expense body schema:")
    for a in actions:
        if a["name"] == "create_travel_expense":
            print(json.dumps(a.get("body", {}), indent=2)[:2000])
            break

    print(f"\nSample: add_per_diem body schema:")
    for a in actions:
        if a["name"] == "add_per_diem":
            print(json.dumps(a.get("body", {}), indent=2)[:1500])
            break

    print(f"\nSample: create_incoming_invoice body schema:")
    for a in actions:
        if a["name"] == "create_incoming_invoice":
            print(json.dumps(a.get("body", {}), indent=2)[:1500])
            break


if __name__ == "__main__":
    main()
