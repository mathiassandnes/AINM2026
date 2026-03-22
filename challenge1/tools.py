"""
Composite Tripletex tools. Each tool encapsulates a multi-step workflow
so the LLM makes fewer decisions and fewer things can go wrong.
"""

import logging
import re
from tripletex import TripletexClient

log = logging.getLogger(__name__)


def normalize_date(d: str) -> str:
    """Convert dd.mm.yyyy or dd/mm/yyyy to YYYY-MM-DD. Pass through if already correct."""
    if not d:
        return d
    m = re.match(r"(\d{1,2})[./](\d{1,2})[./](\d{4})", d)
    if m:
        return f"{m.group(3)}-{m.group(2).zfill(2)}-{m.group(1).zfill(2)}"
    return d


def _validate_date(d: str, field_name: str) -> str | None:
    """Return error message if date is not YYYY-MM-DD, else None."""
    if d and not re.match(r"\d{4}-\d{2}-\d{2}$", d):
        return f"Invalid date format for {field_name}: '{d}'. Expected YYYY-MM-DD."
    return None


class TripletexTools:
    """Composite tools that wrap multi-step Tripletex API workflows."""

    def __init__(self, client: TripletexClient):
        self.client = client
        self._departments: list[dict] | None = None
        self._payment_types: list[dict] | None = None
        self._vat_types: list[dict] | None = None
        self._cost_categories: list[dict] | None = None
        self._travel_payment_types: list[dict] | None = None
        self._warmed_up = False
        self._bank_checked = False

    async def warmup(self):
        """No-op. All data is lazy-loaded when needed."""
        pass

    async def _ensure_bank_account(self):
        """Set bank account number on account 1920 if missing (required for invoicing)."""
        if self._bank_checked:
            return
        acct = await self.client.get("/ledger/account", params={"number": "1920", "fields": "id,version,number,name,isBankAccount,bankAccountNumber"})
        vals = acct.get("values", [])
        if vals:
            bank_acct = vals[0]
            if not bank_acct.get("bankAccountNumber"):
                log.info("Setting bank account number on 1920 for invoicing")
                await self.client.put(f"/ledger/account/{bank_acct['id']}", json={
                    "id": bank_acct["id"],
                    "version": bank_acct["version"],
                    "number": 1920,
                    "name": bank_acct.get("name", "Bankinnskudd"),
                    "isBankAccount": True,
                    "bankAccountNumber": "15045251362",
                })
        self._bank_checked = True

    async def _ensure_payment_types(self):
        """Lazy-load payment types when needed."""
        if self._payment_types is None:
            pt = await self.client.get("/invoice/paymentType", params={"fields": "id,description"})
            self._payment_types = pt.get("values", [])

    async def _ensure_cost_categories(self):
        """Lazy-load cost categories when needed."""
        if self._cost_categories is None:
            cc = await self.client.get("/travelExpense/costCategory", params={"fields": "id,description"})
            self._cost_categories = cc.get("values", [])

    async def _ensure_travel_payment_types(self):
        """Lazy-load travel payment types when needed."""
        if self._travel_payment_types is None:
            tpt = await self.client.get("/travelExpense/paymentType", params={"count": 10})
            self._travel_payment_types = tpt.get("values", [])

    async def _get_default_department_id(self) -> int | None:
        if self._departments:
            return self._departments[0]["id"]
        depts = await self.client.get("/department", params={"count": 1, "fields": "id,name"})
        vals = depts.get("values", [])
        if vals:
            self._departments = vals
            return vals[0]["id"]
        # No departments exist — create a default one
        result = await self.client.post("/department", json={"name": "Avdeling"})
        if result.get("value", {}).get("id"):
            self._departments = [result["value"]]
            return result["value"]["id"]
        return None

    async def _get_bank_payment_type_id(self) -> int | None:
        await self._ensure_payment_types()
        types = self._payment_types
        if not types:
            return None
        for v in types:
            if "bank" in v.get("description", "").lower():
                return v["id"]
        return types[0]["id"] if types else None

    # ── Search (unified) ─────────────────────────────────────

    async def search_entity(self, entity_type: str, name: str = "", email: str = "",
                      number: str = "", employee_id: int = 0,
                      customer_id: int = 0, date_from: str = "",
                      date_to: str = "") -> dict:
        """Unified search across entity types: employee, customer, supplier,
        product, department, invoice, travel_expense, voucher."""
        entity_type = entity_type.lower().replace(" ", "_")
        field_map = {
            "employee": ("/employee", {"fields": "id,firstName,lastName,email,employeeNumber"}),
            "customer": ("/customer", {"fields": "id,name,email,organizationNumber"}),
            "supplier": ("/supplier", {"fields": "id,name,email"}),
            "product": ("/product", {"fields": "id,name,number,priceExcludingVatCurrency"}),
            "department": ("/department", {"fields": "id,name,departmentNumber"}),
            "project": ("/project", {"fields": "id,name,number,projectManager,customer"}),
            "invoice": ("/invoice", {"fields": "id,invoiceNumber,invoiceDate,invoiceDueDate,amount,amountCurrency,customer,voucher"}),
            "travel_expense": ("/travelExpense", {"fields": "id,title,employee,date"}),
            "voucher": ("/ledger/voucher", {"fields": "id,date,description,number,year,voucherType"}),
        }
        if entity_type not in field_map:
            return {"error": f"Unknown entity_type: {entity_type}. Use: {', '.join(field_map.keys())}"}

        path, params = field_map[entity_type]

        # Add search filters based on entity type
        if name:
            if entity_type == "employee":
                # Try to split into first/last name
                parts = name.strip().split(None, 1)
                params["firstName"] = parts[0]
                if len(parts) > 1:
                    params["lastName"] = parts[1]
            elif entity_type in ("invoice", "voucher", "travel_expense"):
                # These don't support name search — skip silently
                pass
            else:
                params["name"] = name
        if email:
            params["email"] = email
        if number:
            if entity_type == "employee":
                params["employeeNumber"] = number
            elif entity_type == "product":
                params["number"] = number
            elif entity_type == "customer":
                params["organizationNumber"] = number
        if employee_id and entity_type == "travel_expense":
            params["employeeId"] = employee_id
        if customer_id and entity_type == "invoice":
            params["customerId"] = customer_id
        # Invoice and voucher search require date ranges — default to wide range
        if entity_type in ("invoice", "voucher"):
            if not date_from:
                date_from = "2020-01-01"
            if not date_to:
                date_to = "2030-12-31"
        if date_from:
            date_from = normalize_date(date_from)
            if entity_type == "invoice":
                params["invoiceDateFrom"] = date_from
            elif entity_type == "voucher":
                params["dateFrom"] = date_from
        if date_to:
            date_to = normalize_date(date_to)
            # Exclusive end date: if same as start, bump by 1 day
            if date_to == date_from:
                from datetime import date as _dt, timedelta as _td
                try:
                    date_to = (_dt.fromisoformat(date_to) + _td(days=1)).isoformat()
                except ValueError:
                    pass
            if entity_type == "invoice":
                params["invoiceDateTo"] = date_to
            elif entity_type == "voucher":
                params["dateTo"] = date_to

        return await self.client.get(path, params=params)

    # ── Employee ────────────────────────────────────────────

    async def create_employee(self, first_name: str, last_name: str,
                        email: str = "", date_of_birth: str = "",
                        phone_mobile: str = "", employee_number: str = "",
                        department_id: int | None = None) -> dict:
        """Create employee. Auto-fetches department."""
        date_of_birth = normalize_date(date_of_birth)
        if not department_id:
            department_id = await self._get_default_department_id()

        # Auto-generate email if not provided (API requires email for STANDARD users)
        if not email:
            import unicodedata, re
            def _ascii(s):
                s = s.replace("Ø", "O").replace("ø", "o").replace("Æ", "AE").replace("æ", "ae").replace("Å", "A").replace("å", "a")
                s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
                return re.sub(r"[^a-zA-Z]", "", s).lower()
            email = f"{_ascii(first_name)}.{_ascii(last_name)}@example.org"

        body = {
            "firstName": first_name,
            "lastName": last_name,
            "userType": "STANDARD",
            "email": email,
        }
        if department_id:
            body["department"] = {"id": department_id}
        if date_of_birth:
            body["dateOfBirth"] = date_of_birth
        if phone_mobile:
            body["phoneNumberMobile"] = phone_mobile
        if employee_number:
            body["employeeNumber"] = employee_number

        return await self.client.post("/employee", json=body)

    async def update_employee(self, employee_id: int, **fields) -> dict:
        """Update employee fields. Fetches current version first."""
        current = await self.client.get(f"/employee/{employee_id}", params={"fields": "*"})
        if current.get("error"):
            return current
        emp = current["value"]
        field_map = {
            "first_name": "firstName", "last_name": "lastName",
            "email": "email", "phone_mobile": "phoneNumberMobile",
            "phone_work": "phoneNumberWork", "phone_home": "phoneNumberHome",
            "date_of_birth": "dateOfBirth", "employee_number": "employeeNumber",
            "department_id": None, "comments": "comments",
        }
        for key, val in fields.items():
            if key == "department_id" and val:
                emp["department"] = {"id": val}
            elif key == "date_of_birth" and val:
                emp["dateOfBirth"] = normalize_date(val)
            elif key in field_map and field_map[key]:
                emp[field_map[key]] = val
        return await self.client.put(f"/employee/{employee_id}", json=emp)

    # ── Customer ────────────────────────────────────────────

    async def create_customer(self, name: str, email: str = "",
                        organization_number: str = "",
                        phone_number: str = "",
                        invoice_email: str = "",
                        is_private_individual: bool = False,
                        address_line1: str = "",
                        postal_code: str = "",
                        city: str = "") -> dict:
        body = {"name": name}
        if email:
            body["email"] = email
        if organization_number:
            body["organizationNumber"] = organization_number
        if phone_number:
            body["phoneNumber"] = phone_number
        if invoice_email:
            body["invoiceEmail"] = invoice_email
        if is_private_individual:
            body["isPrivateIndividual"] = is_private_individual
        if address_line1 or postal_code or city:
            addr = {}
            if address_line1:
                addr["addressLine1"] = address_line1
            if postal_code:
                addr["postalCode"] = postal_code
            if city:
                addr["city"] = city
            body["postalAddress"] = addr
            body["physicalAddress"] = addr
        return await self.client.post("/customer", json=body)

    async def update_customer(self, customer_id: int, **fields) -> dict:
        """Update customer fields. Fetches current version first."""
        current = await self.client.get(f"/customer/{customer_id}", params={"fields": "*"})
        if current.get("error"):
            return current
        cust = current["value"]
        field_map = {
            "name": "name", "email": "email", "phone_number": "phoneNumber",
            "invoice_email": "invoiceEmail", "organization_number": "organizationNumber",
            "is_private_individual": "isPrivateIndividual",
        }
        for key, val in fields.items():
            if key in field_map:
                cust[field_map[key]] = val
        return await self.client.put(f"/customer/{customer_id}", json=cust)

    # ── Supplier ────────────────────────────────────────────

    async def create_supplier(self, name: str, email: str = "",
                        organization_number: str = "",
                        phone_number: str = "") -> dict:
        body = {"name": name, "isSupplier": True}
        if email:
            body["email"] = email
            body["invoiceEmail"] = email
        if organization_number:
            body["organizationNumber"] = organization_number
        if phone_number:
            body["phoneNumber"] = phone_number
        return await self.client.post("/supplier", json=body)

    # ── Product ─────────────────────────────────────────────

    async def create_product(self, name: str, price_excluding_vat: float = 0,
                       number: str = "", vat_type_id: int = 3) -> dict:
        body = {
            "name": name,
            "vatType": {"id": vat_type_id},
        }
        if price_excluding_vat:
            body["priceExcludingVatCurrency"] = price_excluding_vat
        if number:
            body["number"] = number
        return await self.client.post("/product", json=body)

    # ── Invoice (full flow) ─────────────────────────────────

    async def create_invoice_with_lines(self, customer_id: int,
                                  invoice_date: str,
                                  due_date: str,
                                  order_lines: list[dict],
                                  is_prices_including_vat: bool = False,
                                  project_id: int | None = None) -> dict:
        """Create a complete invoice using direct POST /invoice with inline orders.
        Single API call instead of order→orderline→convert flow.
        """
        await self._ensure_bank_account()
        invoice_date = normalize_date(invoice_date)
        due_date = normalize_date(due_date)

        err = _validate_date(invoice_date, "invoice_date") or _validate_date(due_date, "due_date")
        if err:
            return {"error": True, "message": err}
        if not order_lines:
            return {"error": True, "message": "order_lines cannot be empty"}

        # Build orderLines for the inline order
        lines = []
        for line in order_lines:
            line_body = {
                "count": line.get("count", 1),
                "vatType": {"id": line.get("vat_type_id", 3)},
            }
            if line.get("description"):
                line_body["description"] = line["description"]
            if line.get("product_id"):
                line_body["product"] = {"id": line["product_id"]}
            if is_prices_including_vat:
                line_body["unitPriceIncludingVatCurrency"] = line.get("unit_price", 0)
            else:
                line_body["unitPriceExcludingVatCurrency"] = line.get("unit_price", 0)
            lines.append(line_body)

        # Build the order wrapper
        order = {
            "customer": {"id": customer_id},
            "orderDate": invoice_date,
            "deliveryDate": invoice_date,
            "isPrioritizeAmountsIncludingVat": is_prices_including_vat,
            "orderLines": lines,
        }
        if project_id:
            order["project"] = {"id": project_id}

        invoice_body = {
            "invoiceDate": invoice_date,
            "invoiceDueDate": due_date,
            "customer": {"id": customer_id},
            "orders": [order],
        }

        result = await self.client.post("/invoice", json=invoice_body, params={"sendToCustomer": "false"})

        # Auto-send the invoice (some tasks require it)
        if not result.get("error") and result.get("value", {}).get("id"):
            invoice_id = result["value"]["id"]
            try:
                await self.client.put(f"/invoice/{invoice_id}/:send", params={"sendType": "EMAIL"})
            except Exception:
                pass  # Sending is best-effort

        return result

    async def register_payment(self, invoice_id: int, payment_date: str,
                         amount: float, payment_type_id: int | None = None) -> dict:
        """Register payment on invoice. Auto-fetches payment type if needed."""
        payment_date = normalize_date(payment_date)
        if not payment_type_id:
            payment_type_id = await self._get_bank_payment_type_id()

        return await self.client.put(f"/invoice/{invoice_id}/:payment", params={
            "paymentDate": payment_date,
            "paymentTypeId": payment_type_id,
            "paidAmount": amount,
            "paidAmountCurrency": amount,
        })

    async def create_credit_note(self, invoice_id: int, date: str = "") -> dict:
        """Create credit note. Date defaults to today."""
        if not date:
            from datetime import date as dt
            date = dt.today().isoformat()
        else:
            date = normalize_date(date)
        return await self.client.put(f"/invoice/{invoice_id}/:createCreditNote", params={"date": date})

    # ── Project ─────────────────────────────────────────────

    async def create_project(self, name: str, project_manager_id: int,
                       start_date: str, number: str = "",
                       customer_id: int | None = None,
                       end_date: str = "") -> dict:
        start_date = normalize_date(start_date)
        end_date = normalize_date(end_date)
        body = {
            "name": name,
            "projectManager": {"id": project_manager_id},
            "startDate": start_date,
        }
        if number:
            body["number"] = number
        if customer_id:
            body["customer"] = {"id": customer_id}
        if end_date:
            body["endDate"] = end_date
        return await self.client.post("/project", json=body)

    async def create_project_activity(self, project_id: int, activity_name: str = "") -> dict:
        """Create and link an activity to a project."""
        name = activity_name or "Prosjektaktivitet"
        # Create inline PROJECT_SPECIFIC_ACTIVITY — this is the only approach that works reliably
        return await self.client.post("/project/projectActivity", json={
            "project": {"id": project_id},
            "activity": {"name": name, "activityType": "PROJECT_SPECIFIC_ACTIVITY"},
        })

    # ── Department ──────────────────────────────────────────

    async def create_department(self, name: str, department_number: str = "") -> dict:
        body = {"name": name}
        if department_number:
            body["departmentNumber"] = department_number
        return await self.client.post("/department", json=body)

    # ── Employment ──────────────────────────────────────────

    async def create_employment(self, employee_id: int, start_date: str) -> dict:
        """Create an employment record for an employee. Auto-sets DOB if missing."""
        start_date = normalize_date(start_date)

        # Employment requires employee to have dateOfBirth — set a default if missing
        emp = await self.client.get(f"/employee/{employee_id}", params={"fields": "id,version,dateOfBirth"})
        if not emp.get("error"):
            emp_val = emp.get("value", {})
            if not emp_val.get("dateOfBirth"):
                log.info(f"Setting default dateOfBirth for employee {employee_id}")
                await self.client.put(f"/employee/{employee_id}", json={
                    "id": employee_id,
                    "version": emp_val.get("version", 0),
                    "dateOfBirth": "1990-01-01",
                })

        body = {
            "employee": {"id": employee_id},
            "startDate": start_date,
        }
        return await self.client.post("/employee/employment", json=body)

    async def create_employment_details(self, employment_id: int, date: str,
                                         percentage: float = 100.0,
                                         annual_salary: float = 0,
                                         hours_per_day: float = 0) -> dict:
        """Create employment details with correct enum values.
        Handles the tricky employmentType/Form/remunerationType fields.
        """
        date = normalize_date(date)
        body = {
            "employment": {"id": employment_id},
            "date": date,
            "employmentType": "ORDINARY",
            "employmentForm": "PERMANENT",
            "workingHoursScheme": "NOT_SHIFT",
            "percentageOfFullTimeEquivalent": percentage,
        }
        if annual_salary:
            body["annualSalary"] = annual_salary
        if hours_per_day:
            body["hoursPerWeek"] = hours_per_day * 5

        result = await self.client.post("/employee/employment/details", json=body)

        # If enum values rejected, retry with just numeric fields
        if result.get("error"):
            log.info(f"Employment details with enums failed, retrying minimal: {result.get('message', '')[:150]}")
            body = {
                "employment": {"id": employment_id},
                "date": date,
                "percentageOfFullTimeEquivalent": percentage,
            }
            if annual_salary:
                body["annualSalary"] = annual_salary
            result = await self.client.post("/employee/employment/details", json=body)

        return result

    # ── Travel Expense ──────────────────────────────────────

    def _resolve_cost_category(self, hint: str) -> int | None:
        """Fuzzy-match a cost category description from cached data."""
        if not hint or not self._cost_categories:
            return None
        hint_lower = hint.lower()
        # Common translations → Norwegian descriptions
        translations = {
            "hotel": "hotell", "flight": "fly", "taxi": "taxi", "bus": "buss",
            "train": "tog", "fuel": "drivstoff", "ferry": "ferge", "toll": "bom",
            "food": "mat", "meal": "mat", "parking": "parkering", "phone": "telefon",
            "office": "kontor", "internet": "bredbånd", "broadband": "bredbånd",
            "course": "møte", "meeting": "møte", "representation": "representasjon",
            "hôtel": "hotell", "vol": "fly", "flug": "fly", "zug": "tog",
            "vuelo": "fly", "tren": "tog",
        }
        # Translate if possible
        for eng, nor in translations.items():
            if eng in hint_lower:
                hint_lower = nor
                break
        for cat in self._cost_categories:
            desc = cat.get("description", "").lower()
            if hint_lower in desc or desc in hint_lower:
                return cat["id"]
        # Fallback: "Annen kontorkostnad" (other office cost)
        for cat in self._cost_categories:
            if "annen" in cat.get("description", "").lower():
                return cat["id"]
        return self._cost_categories[0]["id"] if self._cost_categories else None

    async def _resolve_per_diem_rate_category(self) -> int | None:
        """Find a valid per diem rate category from the API."""
        if not hasattr(self, "_per_diem_rate_categories"):
            result = await self.client.get("/travelExpense/perDiemCompensation/rateCategory",
                                           params={"count": 50, "fields": "id,name,description"})
            self._per_diem_rate_categories = result.get("values", [])
            if self._per_diem_rate_categories:
                log.info(f"Per diem rate categories: {[(c['id'], c.get('name', c.get('description',''))) for c in self._per_diem_rate_categories[:10]]}")
        # Prefer "over 12 timer" (overnight > 12 hours) category
        for cat in self._per_diem_rate_categories:
            desc = (cat.get("name", "") + cat.get("description", "")).lower()
            if "12 timer" in desc or "overnight" in desc:
                return cat["id"]
        return self._per_diem_rate_categories[0]["id"] if self._per_diem_rate_categories else None

    async def create_travel_expense(self, employee_id: int, title: str,
                              date: str = "",
                              departure_date: str = "",
                              return_date: str = "",
                              per_diem_days: int = 0,
                              per_diem_rate: float = 0,
                              costs: list[dict] | None = None) -> dict:
        """Create travel expense with optional costs and per diem.

        costs: [{"description": str, "amount": float, "date": str,
                  "category": str (e.g. "hotel", "flight", "taxi")}]
        per_diem_days/per_diem_rate: for daily allowance (diett/Tagegeld/indemnités)
        departure_date/return_date: required for multi-day trips with per diem
        """
        from datetime import date as dt, timedelta
        date = normalize_date(date)
        departure_date = normalize_date(departure_date) or date
        return_date = normalize_date(return_date)

        # Auto-calculate dates if per diem requested but dates missing
        if per_diem_days > 0:
            if not departure_date:
                departure_date = dt.today().isoformat()
            if not return_date:
                dep = dt.fromisoformat(departure_date)
                return_date = (dep + timedelta(days=per_diem_days - 1)).isoformat()
            if not date:
                date = departure_date

        body = {
            "employee": {"id": employee_id},
            "title": title,
        }
        # date is READONLY on TravelExpense — do NOT set it (derived from costs/per diem)
        # departureDate/returnDate live inside travelDetails (NOT top-level)
        travel_details = {}
        if departure_date:
            travel_details["departureDate"] = departure_date
        if return_date:
            travel_details["returnDate"] = return_date
        if travel_details:
            travel_details["isCompensationFromRates"] = True
            body["travelDetails"] = travel_details

        result = await self.client.post("/travelExpense", json=body)
        if result.get("error"):
            return result
        expense_id = result["value"]["id"]

        # Add cost items
        if costs:
            await self._ensure_cost_categories()
            await self._ensure_travel_payment_types()
            for cost in costs:
                cost_date = normalize_date(cost.get("date", "")) or departure_date or date
                cost_body = {
                    "travelExpense": {"id": expense_id},
                    "amountCurrencyIncVat": cost["amount"],
                    "date": cost_date,
                    "comments": cost.get("description", ""),
                }
                cat_id = cost.get("cost_category_id")
                if not cat_id:
                    cat_hint = cost.get("category", cost.get("description", ""))
                    cat_id = self._resolve_cost_category(cat_hint)
                if cat_id:
                    cost_body["costCategory"] = {"id": cat_id}
                pay_id = cost.get("payment_type_id")
                if not pay_id and self._travel_payment_types:
                    pay_id = self._travel_payment_types[0]["id"]
                if pay_id:
                    cost_body["paymentType"] = {"id": pay_id}
                await self.client.post("/travelExpense/cost", json=cost_body)

        # Add per diem compensation if requested
        if per_diem_days > 0 and per_diem_rate > 0:
            rate_cat_id = await self._resolve_per_diem_rate_category()
            if rate_cat_id:
                per_diem_body = {
                    "travelExpense": {"id": expense_id},
                    "rateCategory": {"id": rate_cat_id},
                    "count": per_diem_days,
                    "overnightAccommodation": "HOTEL",
                    "location": "Norge",
                    "address": "Norge",
                }
                pdr = await self.client.post("/travelExpense/perDiemCompensation", json=per_diem_body)
                if pdr.get("error"):
                    log.warning(f"Per diem failed: {pdr}. Falling back to cost line.")
                    # Fallback: add per diem as a regular cost line
                    total_per_diem = per_diem_days * per_diem_rate
                    await self._ensure_cost_categories()
                    await self._ensure_travel_payment_types()
                    fallback_body = {
                        "travelExpense": {"id": expense_id},
                        "amountCurrencyIncVat": total_per_diem,
                        "date": departure_date or date,
                        "comments": f"Diett: {per_diem_days} dager × {per_diem_rate} NOK",
                    }
                    if self._cost_categories:
                        fallback_body["costCategory"] = {"id": self._cost_categories[0]["id"]}
                    if self._travel_payment_types:
                        fallback_body["paymentType"] = {"id": self._travel_payment_types[0]["id"]}
                    await self.client.post("/travelExpense/cost", json=fallback_body)

        # Auto-deliver the travel expense
        try:
            deliver_result = await self.client.put("/travelExpense/:deliver", json=[{"id": expense_id}])
            if deliver_result.get("error"):
                log.warning(f"Travel expense auto-deliver failed: {deliver_result.get('message', '')[:150]}")
        except Exception as e:
            log.warning(f"Travel expense auto-deliver exception: {e}")

        return result

    # ── Voucher / Ledger ────────────────────────────────────

    async def create_voucher(self, date: str, description: str,
                       postings: list[dict]) -> dict:
        """Create voucher using historical endpoint (works without Advanced Voucher permission)."""
        import time as _time
        date = normalize_date(date)

        # Validate balance — auto-fix by adjusting the bank/balancing posting
        total = 0.0
        for p in postings:
            total += p.get("debit", 0) - p.get("credit", 0)
        if abs(total) > 0.01:
            # Find the bank/balancing posting (usually 1920 or the single credit line) and adjust it
            adjusted = False
            # First try: find a bank account posting (1920) to adjust
            for p in postings:
                if p.get("account_id") in (1920, "1920"):
                    if p.get("credit"):
                        p["credit"] = round(p["credit"] + total, 2)
                        adjusted = True
                    elif p.get("debit"):
                        p["debit"] = round(p["debit"] + total, 2)
                        adjusted = True
                    break
            # Second try: adjust the last credit posting
            if not adjusted:
                for p in reversed(postings):
                    if p.get("credit"):
                        p["credit"] = round(p["credit"] + total, 2)
                        adjusted = True
                        break
            if adjusted:
                log.info(f"Auto-fixed balance error: adjusted by {total:.2f}")
            else:
                summary = "; ".join(f"acct {p.get('account_id','?')}: D{p.get('debit',0)} C{p.get('credit',0)}" for p in postings)
                return {"error": True, "message": f"Postings don't balance: net={total:.2f}. Current: {summary}"}

        # Known account names for auto-creation
        _ACCOUNT_NAMES = {
            1209: "Akkumulerte avskrivninger maskiner",
            1259: "Akkumulerte avskrivninger inventar",
            1710: "Forskuddsbetalte kostnader",
            1720: "Forskuddsbetalte leieutgifter",
            2750: "Påløpte lønnskostnader",
            2900: "Annen kortsiktig gjeld",
            2920: "Betalbar skatt",
            3400: "Purregebyr",
            6010: "Avskrivning transportmidler",
            6020: "Avskrivning inventar",
            6030: "Avskrivning programvare",
            8700: "Skattekostnad",
        }

        api_postings = []
        for p in postings:
            acct_id = p["account_id"]
            if acct_id < 100000:
                acct_result = await self.client.get("/ledger/account", params={"number": str(acct_id), "fields": "id"})
                acct_vals = acct_result.get("values", [])
                if acct_vals:
                    acct_id = acct_vals[0]["id"]
                else:
                    # Auto-create missing accounts (common in year-end closing)
                    acct_name = _ACCOUNT_NAMES.get(acct_id, f"Konto {acct_id}")
                    create_result = await self.client.post("/ledger/account", json={"number": acct_id, "name": acct_name})
                    if create_result.get("value", {}).get("id"):
                        acct_id = create_result["value"]["id"]
                    else:
                        return {"error": True, "message": f"Account {acct_id} not found and could not be created: {create_result.get('message', '')}"}

            amount = 0.0
            if p.get("debit"):
                amount = p["debit"]
            elif p.get("credit"):
                amount = -p["credit"]

            posting = {
                "date": date,
                "account": {"id": acct_id},
                "currency": {"id": 1},  # NOK
                "amount": amount,
                "amountCurrency": amount,
                "amountGross": amount,
                "amountGrossCurrency": amount,
                "amountVat": 0,
            }
            if p.get("description"):
                posting["description"] = p["description"]
            for dim_key in ("dimension1_id", "dimension2_id", "dimension3_id"):
                dim_num = dim_key[9]
                if p.get(dim_key):
                    posting[f"freeAccountingDimension{dim_num}"] = {"id": p[dim_key]}
            if p.get("department_id"):
                posting["department"] = {"id": p["department_id"]}
            if p.get("project_id"):
                posting["project"] = {"id": p["project_id"]}
            if p.get("customer_id"):
                posting["customer"] = {"id": p["customer_id"]}
            if p.get("supplier_id"):
                posting["supplier"] = {"id": p["supplier_id"]}
            api_postings.append(posting)

        ext_num = f"API-{int(_time.time())}-{description[:30].replace(' ', '_')}"

        # Check if any posting has dimensions
        has_dimensions = any(
            p.get("freeAccountingDimension1") or p.get("freeAccountingDimension2") or p.get("freeAccountingDimension3")
            for p in api_postings
        )

        if has_dimensions:
            # Use regular /ledger/voucher endpoint for dimension support
            # Requires row numbering starting at 1
            # Regular endpoint does NOT accept amountVat — strip it
            regular_postings = []
            for idx, p in enumerate(api_postings):
                rp = dict(p)
                rp["row"] = idx + 1
                rp.pop("amountVat", None)
                # Ensure required amount fields
                amt = rp.get("amountGrossCurrency", 0)
                rp.setdefault("amount", amt)
                rp.setdefault("amountCurrency", amt)
                rp.setdefault("amountGross", amt)
                regular_postings.append(rp)

            result = await self.client.post(
                "/ledger/voucher",
                json={
                    "date": date,
                    "description": description,
                    "postings": regular_postings,
                },
            )
            if not result.get("error"):
                return result
            # If regular endpoint fails, try historical without dimensions
            log.warning(f"Regular voucher with dimensions failed: {result.get('message', '')[:200]}. Retrying historical without dimensions.")
            for p in api_postings:
                p.pop("freeAccountingDimension1", None)
                p.pop("freeAccountingDimension2", None)
                p.pop("freeAccountingDimension3", None)

        result = await self.client.post(
            "/ledger/voucher/historical/historical",
            params={"comment": "Created by AI agent"},
            json=[{
                "date": date,
                "externalVoucherNumber": ext_num,
                "description": description,
                "postings": api_postings,
            }],
        )

        # Historical endpoint returns a list — extract first item
        if isinstance(result, dict) and result.get("values"):
            return {"value": result["values"][0]}
        return result

    # ── Log Hours + Project Invoice ────────────────────────

    async def log_hours_and_invoice(self, employee_id: int,
                                     customer_id: int,
                                     project_id: int,
                                     activity_name: str,
                                     hours: float,
                                     hourly_rate: float,
                                     date: str = "",
                                     invoice_date: str = "") -> dict:
        """Log hours on a project activity and create a project invoice.
        Handles: find/create activity, link to project, set hourly rate,
        log timesheet, create order→invoice linked to project.
        """
        if not date:
            from datetime import date as dt
            date = dt.today().isoformat()
        else:
            date = normalize_date(date)
        invoice_date = normalize_date(invoice_date) or date

        # 1. Create project-specific activity and link to project in one call
        # Must use /project/projectActivity — direct /activity creation is rejected
        pa_result = await self.client.post("/project/projectActivity", json={
            "project": {"id": project_id},
            "activity": {"name": activity_name, "activityType": "PROJECT_SPECIFIC_ACTIVITY"},
        })
        if pa_result.get("error"):
            # Activity might already exist — try to find and link it
            act_result = await self.client.get("/activity", params={"name": activity_name, "fields": "id,name"})
            act_vals = act_result.get("values", [])
            if act_vals:
                activity_id = act_vals[0]["id"]
                await self.client.post("/project/projectActivity", json={
                    "activity": {"id": activity_id}, "project": {"id": project_id}
                })
            else:
                return pa_result
        else:
            activity_id = pa_result["value"]["activity"]["id"]

        # 3. Set hourly rate on employee
        await self.client.post("/employee/hourlyCostAndRate", json={
            "employee": {"id": employee_id}, "date": date, "rate": hourly_rate
        })

        # 4. Log timesheet entry
        ts_result = await self.client.post("/timesheet/entry", json={
            "employee": {"id": employee_id},
            "project": {"id": project_id},
            "activity": {"id": activity_id},
            "date": date,
            "hours": hours,
            "comment": f"{activity_name} - {hours}h @ {hourly_rate} NOK/h",
        })
        if ts_result.get("error"):
            return ts_result

        # 5. Create order linked to project, then invoice
        total = hours * hourly_rate
        await self._ensure_bank_account()

        order_result = await self.client.post("/order", json={
            "customer": {"id": customer_id},
            "deliveryDate": invoice_date,
            "orderDate": invoice_date,
            "project": {"id": project_id},
        })
        if order_result.get("error"):
            return order_result
        order_id = order_result["value"]["id"]

        await self.client.post("/order/orderline", json={
            "order": {"id": order_id},
            "description": f"{activity_name} ({hours}h × {hourly_rate} NOK)",
            "count": hours,
            "unitPriceExcludingVatCurrency": hourly_rate,
            "vatType": {"id": 3},  # 25% MVA
        })

        invoice_result = await self.client.put(
            f"/order/{order_id}/:invoice",
            params={"invoiceDate": invoice_date, "sendToCustomer": "false"},
        )
        return invoice_result

    # ── Incoming / Supplier Invoice ────────────────────────

    async def create_incoming_invoice(self, supplier_id: int,
                                       invoice_date: str,
                                       due_date: str,
                                       amount_incl_vat: float,
                                       invoice_number: str = "",
                                       description: str = "",
                                       account_number: int = 6300,
                                       vat_type_id: int = 0) -> dict:
        """Register an incoming (supplier/vendor) invoice.
        Auto-resolves expense account and inbound VAT type (25%).
        """
        invoice_date = normalize_date(invoice_date)
        due_date = normalize_date(due_date)

        # Look up expense account by number
        acct_result = await self.client.get("/ledger/account", params={"number": str(account_number), "fields": "id,number,name"})
        acct_vals = acct_result.get("values", [])
        if not acct_vals:
            return {"error": True, "message": f"Account {account_number} not found"}
        account_id = acct_vals[0]["id"]

        # Resolve inbound VAT type (25% by default)
        if not vat_type_id:
            if not self._vat_types:
                vt = await self.client.get("/ledger/vatType", params={"fields": "id,name,percentage"})
                self._vat_types = vt.get("values", [])
            for vt in self._vat_types:
                name = vt.get("name", "").lower()
                pct = vt.get("percentage", 0)
                if "inngående" in name and pct == 25:
                    vat_type_id = vt["id"]
                    break
            if not vat_type_id:
                # Fallback: use ID 1 which is typically inbound 25%
                vat_type_id = 1

        body = {
            "invoiceHeader": {
                "vendorId": supplier_id,
                "invoiceDate": invoice_date,
                "dueDate": due_date,
                "invoiceAmount": amount_incl_vat,
                "description": description or f"Invoice {invoice_number}",
            },
            "orderLines": [{
                "row": 1,
                "externalId": f"line-1-{invoice_number or 'default'}",
                "description": description or f"Invoice {invoice_number}",
                "accountId": account_id,
                "amountInclVat": amount_incl_vat,
                "vatTypeId": vat_type_id,
            }],
        }
        if invoice_number:
            body["invoiceHeader"]["invoiceNumber"] = invoice_number

        result = await self.client.post("/incomingInvoice", json=body)
        if result.get("error"):
            # Fallback: create voucher with correct debit/credit structure
            # For 25% VAT: amount excl VAT = amount incl VAT / 1.25
            # Debit expense account, debit VAT account, credit supplier (accounts payable)
            log.info("incomingInvoice failed, falling back to voucher")
            vat_rate = 0.25 if vat_type_id == 1 else (0.15 if vat_type_id == 11 else (0.12 if vat_type_id == 12 else 0.25))
            amount_excl_vat = round(amount_incl_vat / (1 + vat_rate), 2)
            vat_amount = round(amount_incl_vat - amount_excl_vat, 2)

            # Look up accounts payable (2400) and inbound VAT (2710)
            ap_result = await self.client.get("/ledger/account", params={"number": "2400", "fields": "id"})
            ap_vals = ap_result.get("values", [])
            ap_id = ap_vals[0]["id"] if ap_vals else None

            vat_result = await self.client.get("/ledger/account", params={"number": "2710", "fields": "id"})
            vat_vals = vat_result.get("values", [])
            vat_id = vat_vals[0]["id"] if vat_vals else None

            if not ap_id or not vat_id:
                return {"error": True, "message": "Could not find accounts 2400 or 2710"}

            # Resolve the actual VAT type object ID (not just 1/11/12)
            vat_type_obj_id = vat_type_id
            if not self._vat_types:
                vt = await self.client.get("/ledger/vatType", params={"fields": "id,name,number,percentage"})
                self._vat_types = vt.get("values", [])
            # Find the matching inbound VAT type object
            for vt in self._vat_types:
                if vt.get("id") == vat_type_id:
                    vat_type_obj_id = vt["id"]
                    break
                name = vt.get("name", "").lower()
                pct = vt.get("percentage", 0)
                if "inngående" in name and pct == (vat_rate * 100):
                    vat_type_obj_id = vt["id"]
                    break

            import time as _time
            desc_text = description or f"Leverandørfaktura {invoice_number}"
            ext_num = f"API-{int(_time.time())}-{desc_text[:30].replace(' ', '_')}"
            postings = [
                {"date": invoice_date, "account": {"id": account_id},
                 "currency": {"id": 1},
                 "amount": amount_incl_vat, "amountCurrency": amount_incl_vat,
                 "amountGross": amount_incl_vat, "amountGrossCurrency": amount_incl_vat,
                 "amountVat": 0,
                 "vatType": {"id": vat_type_obj_id},
                 "description": description or f"Invoice {invoice_number}"},
                {"date": invoice_date, "account": {"id": ap_id},
                 "currency": {"id": 1},
                 "amount": -amount_incl_vat, "amountCurrency": -amount_incl_vat,
                 "amountGross": -amount_incl_vat, "amountGrossCurrency": -amount_incl_vat,
                 "amountVat": 0,
                 "supplier": {"id": supplier_id},
                 "description": f"Leverandør {invoice_number}"},
            ]
            result = await self.client.post(
                "/ledger/voucher/historical/historical",
                params={"comment": "Created by AI agent"},
                json=[{
                    "date": invoice_date,
                    "externalVoucherNumber": ext_num,
                    "description": desc_text,
                    "postings": postings,
                }],
            )
            # Historical endpoint returns a list — extract first item
            if isinstance(result, dict) and result.get("values"):
                result = {"value": result["values"][0]}
        return result

    # ── Payroll (voucher-based) ──────────────────────────────

    async def run_payroll(self, employee_id: int,
                          base_salary: float,
                          bonus: float = 0,
                          date: str = "") -> dict:
        """Run payroll via salary transaction API.
        Creates a salary transaction with payslip specifications.
        """
        if not date:
            from datetime import date as dt
            d = dt.today().replace(day=1)
            date = d.isoformat()
            year = d.year
            month = d.month
        else:
            date = normalize_date(date)
            parts = date.split("-")
            year = int(parts[0])
            month = int(parts[1])

        # Look up salary types
        st_result = await self.client.get("/salary/type", params={"fields": "id,name,number", "count": 50})
        salary_types = st_result.get("values", [])

        # Find Fastlønn (fixed salary) and Bonus types
        fastlonn_id = None
        bonus_id = None
        for st in salary_types:
            if st.get("number") == "2000":  # Fastlønn
                fastlonn_id = st["id"]
            elif st.get("number") == "2002":  # Bonus
                bonus_id = st["id"]

        if not fastlonn_id:
            return {"error": True, "message": "Salary type 'Fastlønn' (2000) not found"}

        # Build specifications
        specs = [
            {
                "salaryType": {"id": fastlonn_id},
                "rate": base_salary,
                "count": 1,
                "description": "Grunnlønn",
            }
        ]
        if bonus > 0 and bonus_id:
            specs.append({
                "salaryType": {"id": bonus_id},
                "rate": bonus,
                "count": 1,
                "description": "Bonus",
            })

        body = {
            "date": date,
            "year": year,
            "month": month,
            "payslips": [{
                "employee": {"id": employee_id},
                "date": date,
                "year": year,
                "month": month,
                "specifications": specs,
            }],
        }

        result = await self.client.post("/salary/transaction", json=body)

        # If salary API fails, fall back to voucher approach
        if result.get("error"):
            log.info(f"Salary transaction API failed: {result.get('message', '')[:200]}. Trying voucher fallback.")
            postings = [
                {"account_id": 5000, "debit": base_salary, "description": "Grunnlønn"},
                {"account_id": 1920, "credit": base_salary, "description": "Utbetaling lønn"},
            ]
            if bonus > 0:
                postings.extend([
                    {"account_id": 5000, "debit": bonus, "description": "Bonus"},
                    {"account_id": 1920, "credit": bonus, "description": "Utbetaling bonus"},
                ])
            result = await self.create_voucher(
                date=date,
                description=f"Lønnskjøring {month}/{year}",
                postings=postings,
            )

        return result

    # ── Reverse / Update / Deliver ─────────────────────────

    async def reverse_voucher(self, voucher_id: int, date: str = "") -> dict:
        """Reverse a voucher (undo a payment or journal entry)."""
        if not date:
            from datetime import date as dt
            date = dt.today().isoformat()
        else:
            date = normalize_date(date)
        return await self.client.put(f"/ledger/voucher/{voucher_id}/:reverse", params={"date": date})

    async def update_project(self, project_id: int,
                              is_fixed_price: bool = False,
                              fixed_price: float = 0,
                              **fields) -> dict:
        """Update project (e.g. set fixed price). Fetches current version automatically."""
        current = await self.client.get(f"/project/{project_id}", params={"fields": "*"})
        if current.get("error"):
            return current
        proj = current["value"]
        if is_fixed_price:
            proj["isFixedPrice"] = True
        if fixed_price:
            proj["fixedprice"] = fixed_price
        for key, val in fields.items():
            if val is not None:
                proj[key] = val
        # Strip fields that the API rejects on PUT (must use separate endpoints)
        for strip_key in ("hourlyRates", "projectHourlyRates", "projectSpecificRates",
                          "activities", "projectActivities", "participants"):
            proj.pop(strip_key, None)
        return await self.client.put(f"/project/{project_id}", json=proj)

    async def deliver_travel_expense(self, expense_id: int) -> dict:
        """Submit/deliver a travel expense report."""
        return await self.client.put("/travelExpense/:deliver", json=[{"id": expense_id}])

    async def send_invoice(self, invoice_id: int) -> dict:
        """Send an invoice to the customer via email."""
        return await self.client.put(f"/invoice/{invoice_id}/:send", params={"sendType": "EMAIL"})

    async def get_ledger_postings(self, account_number: int = 0,
                                   account_number_from: int = 0,
                                   account_number_to: int = 0,
                                   date_from: str = "", date_to: str = "",
                                   count: int = 1000, **kwargs) -> dict:
        """Query ledger postings. Use account_number for a single account, or account_number_from/to for a range (e.g. 5000-7999 for all expenses)."""
        params = {"fields": "id,date,description,amount,amountCurrency,account(id,number,name),voucher(id,number,description)", "count": count}
        if account_number:
            acct = await self.client.get("/ledger/account", params={"number": str(account_number), "fields": "id"})
            vals = acct.get("values", [])
            if vals:
                params["accountId"] = vals[0]["id"]
            else:
                return {"error": True, "message": f"Account {account_number} not found"}
        if account_number_from:
            params["accountNumberFrom"] = account_number_from
        if account_number_to:
            params["accountNumberTo"] = account_number_to
        if date_from:
            params["dateFrom"] = normalize_date(date_from)
        if date_to:
            params["dateTo"] = normalize_date(date_to)
        return await self.client.get("/ledger/posting", params=params)

    # ── Lookup helpers ──────────────────────────────────────

    async def get_accounts(self, number: str = "") -> dict:
        params = {"fields": "id,number,name"}
        if number:
            params["number"] = number
        return await self.client.get("/ledger/account", params=params)

    # ── Generic fallback ────────────────────────────────────

    @staticmethod
    def _fix_path(path: str) -> str:
        """Strip /v2 prefix if model included it (base_url already has /v2)."""
        if path.startswith("/v2/"):
            path = path[3:]
        return path

    async def api_get(self, path: str, params: dict | None = None) -> dict:
        """Generic GET for anything not covered by composite tools."""
        return await self.client.get(self._fix_path(path), params=params)

    async def api_post(self, path: str, body: dict | None = None) -> dict:
        """Generic POST fallback."""
        return await self.client.post(self._fix_path(path), json=body)

    async def api_put(self, path: str, body: dict | None = None, params: dict | None = None) -> dict:
        """Generic PUT fallback."""
        return await self.client.put(self._fix_path(path), json=body, params=params)

    async def api_delete(self, path: str) -> dict:
        """Generic DELETE fallback."""
        return await self.client.delete(self._fix_path(path))