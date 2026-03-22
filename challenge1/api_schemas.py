"""
Tripletex API operation registry.

Each operation defines exact method, path, and JSON Schema for params/body
matching the real API. Used to construct forced structured output tools.
"""


def _ref(description=""):
    """Reference object {id: N} — accepts int or $step_N string ref."""
    schema = {
        "type": "object",
        "properties": {"id": {"oneOf": [{"type": "integer"}, {"type": "string"}]}},
        "required": ["id"],
    }
    if description:
        schema["description"] = description
    return schema


def _id_or_ref(description=""):
    """ID field that accepts int or $step_N string ref."""
    schema = {"oneOf": [{"type": "integer"}, {"type": "string"}]}
    if description:
        schema["description"] = description
    return schema


def _num_or_ref(description=""):
    """Number field that accepts number or $step_N string ref."""
    schema = {"oneOf": [{"type": "number"}, {"type": "string"}]}
    if description:
        schema["description"] = description
    return schema


# ── Operation Registry ────────────────────────────────────────────
# Keys are operation IDs used in Phase 1. Each operation defines:
#   method, path, description
#   body_schema: JSON Schema for POST/PUT body (None for GET/DELETE)
#   params_schema: JSON Schema for query params (None if no params)
#   default_params: always-included query params (e.g. fields filter)
#   needs_bank_setup: True if this operation requires bank account to be configured

OPERATIONS = {
    # ── Lookups (GET) ─────────────────────────────────────
    "get_department": {
        "method": "GET",
        "path": "/department",
        "description": "Find department (usually to get default dept ID)",
        "default_params": {"count": 1, "fields": "id,name"},
        "params_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Filter by name"},
            },
        },
    },
    "find_employee": {
        "method": "GET",
        "path": "/employee",
        "description": "Find employee by name/email",
        "default_params": {"fields": "id,firstName,lastName,email,employeeNumber"},
        "params_schema": {
            "type": "object",
            "properties": {
                "firstName": {"type": "string"},
                "lastName": {"type": "string"},
                "email": {"type": "string"},
                "employeeNumber": {"type": "string"},
            },
        },
    },
    "find_customer": {
        "method": "GET",
        "path": "/customer",
        "description": "Find customer by name",
        "default_params": {"fields": "id,name,email,organizationNumber"},
        "params_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"},
                "organizationNumber": {"type": "string"},
            },
        },
    },
    "find_supplier": {
        "method": "GET",
        "path": "/supplier",
        "description": "Find supplier by name",
        "default_params": {"fields": "id,name,email"},
        "params_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
        },
    },
    "find_product": {
        "method": "GET",
        "path": "/product",
        "description": "Find product by name or number",
        "default_params": {"fields": "id,name,number,priceExcludingVatCurrency"},
        "params_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "number": {"type": "string"},
            },
        },
    },
    "find_project": {
        "method": "GET",
        "path": "/project",
        "description": "Find project by name",
        "default_params": {"fields": "id,name,number,projectManager,customer,version"},
        "params_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
        },
    },
    "find_invoice": {
        "method": "GET",
        "path": "/invoice",
        "description": "Find invoice by customer/date. invoiceDateFrom and invoiceDateTo are REQUIRED by the API.",
        "default_params": {"fields": "id,invoiceNumber,amount,customer,voucher"},
        "params_schema": {
            "type": "object",
            "properties": {
                "customerId": _id_or_ref("Customer ID"),
                "invoiceDateFrom": {"type": "string", "description": "YYYY-MM-DD (REQUIRED)"},
                "invoiceDateTo": {"type": "string", "description": "YYYY-MM-DD (REQUIRED)"},
            },
            "required": ["invoiceDateFrom", "invoiceDateTo"],
        },
    },
    "find_voucher": {
        "method": "GET",
        "path": "/ledger/voucher",
        "description": "Find voucher",
        "default_params": {"fields": "id,date,description,number"},
        "params_schema": {
            "type": "object",
            "properties": {
                "dateFrom": {"type": "string"},
                "dateTo": {"type": "string"},
                "description": {"type": "string"},
            },
        },
    },
    "find_activity": {
        "method": "GET",
        "path": "/activity",
        "description": "Find activity by name",
        "default_params": {"fields": "id,name"},
        "params_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
        },
    },
    "get_payment_types": {
        "method": "GET",
        "path": "/invoice/paymentType",
        "description": "Get invoice payment types (find 'bank' type)",
        "default_params": {"fields": "id,description"},
        "params_schema": None,
    },
    "get_cost_categories": {
        "method": "GET",
        "path": "/travelExpense/costCategory",
        "description": "Get travel cost categories (Fly, Hotell, Taxi, etc.)",
        "default_params": {"count": 100},
        "params_schema": None,
    },
    "get_travel_payment_types": {
        "method": "GET",
        "path": "/travelExpense/paymentType",
        "description": "Get travel expense payment types",
        "default_params": {"count": 10},
        "params_schema": None,
    },
    "get_rate_categories": {
        "method": "GET",
        "path": "/travelExpense/rateCategory",
        "description": "Get per diem rate categories (id=11 for domestic overnight)",
        "default_params": {"count": 100},
        "params_schema": None,
    },
    "get_salary_types": {
        "method": "GET",
        "path": "/salary/type",
        "description": "Get salary types (Fastlønn=2000, Bonus=2002)",
        "default_params": {"fields": "id,name,number"},
        "params_schema": None,
    },
    "create_account": {
        "method": "POST",
        "path": "/ledger/account",
        "description": "Create a ledger account (e.g. 1209 for accumulated depreciation)",
        "body_schema": {
            "type": "object",
            "properties": {
                "number": {"type": "integer", "description": "Account number e.g. 1209"},
                "name": {"type": "string", "description": "Account name in Norwegian"},
            },
            "required": ["number", "name"],
        },
    },
    "get_account": {
        "method": "GET",
        "path": "/ledger/account",
        "description": "Find account by number (e.g. 1920, 6300)",
        "default_params": {"fields": "id,number,name,version,isBankAccount,bankAccountNumber"},
        "params_schema": {
            "type": "object",
            "properties": {
                "number": {"type": "string", "description": "Account number e.g. '1920'"},
            },
            "required": ["number"],
        },
    },

    # ── Creates (POST) ────────────────────────────────────
    "create_employee": {
        "method": "POST",
        "path": "/employee",
        "description": "Create employee",
        "body_schema": {
            "type": "object",
            "properties": {
                "firstName": {"type": "string"},
                "lastName": {"type": "string"},
                "email": {"type": "string", "description": "REQUIRED for STANDARD users. Extract from PDF/prompt or generate firstname.lastname@example.org"},
                "dateOfBirth": {"type": "string", "description": "YYYY-MM-DD"},
                "phoneNumberMobile": {"type": "string"},
                "employeeNumber": {"type": "string"},
                "userType": {"type": "string", "enum": ["STANDARD", "EXTENDED", "NO_ACCESS"]},
                "department": _ref("Department — use $step_N.values[0].id from get_department, or $step_N.value.id from create_department"),
            },
            "required": ["firstName", "lastName", "email", "userType", "department"],
        },
    },
    "create_customer": {
        "method": "POST",
        "path": "/customer",
        "description": "Create customer",
        "body_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"},
                "isCustomer": {"type": "boolean", "default": True},
                "phoneNumber": {"type": "string"},
                "invoiceEmail": {"type": "string"},
                "organizationNumber": {"type": "string"},
                "isPrivateIndividual": {"type": "boolean"},
                "postalAddress": {
                    "type": "object",
                    "properties": {
                        "addressLine1": {"type": "string"},
                        "postalCode": {"type": "string"},
                        "city": {"type": "string"},
                    },
                },
            },
            "required": ["name"],
        },
    },
    "create_supplier": {
        "method": "POST",
        "path": "/supplier",
        "description": "Create supplier",
        "body_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"},
                "isSupplier": {"type": "boolean", "default": True},
                "phoneNumber": {"type": "string"},
                "organizationNumber": {"type": "string"},
            },
            "required": ["name"],
        },
    },
    "create_product": {
        "method": "POST",
        "path": "/product",
        "description": "Create product",
        "body_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "number": {"type": "string", "description": "Product number"},
                "priceExcludingVatCurrency": {"type": "number"},
                "vatType": _ref("VAT type: 3=25%, 31=15% food, 33=12%, 5=0% newspapers, 6=0% exempt"),
            },
            "required": ["name"],
        },
    },
    "create_department": {
        "method": "POST",
        "path": "/department",
        "description": "Create department",
        "body_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "departmentNumber": {"type": "string"},
            },
            "required": ["name"],
        },
    },
    "create_invoice_direct": {
        "method": "POST",
        "path": "/invoice",
        "description": "Create invoice directly with inline orders and orderLines. Single API call.",
        "body_schema": {
            "type": "object",
            "properties": {
                "invoiceDate": {"type": "string", "description": "YYYY-MM-DD"},
                "invoiceDueDate": {"type": "string", "description": "YYYY-MM-DD"},
                "customer": _ref("Customer"),
                "orders": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "customer": _ref("Customer (same as invoice customer)"),
                            "orderDate": {"type": "string", "description": "YYYY-MM-DD (same as invoiceDate)"},
                            "deliveryDate": {"type": "string", "description": "YYYY-MM-DD"},
                            "project": _ref("Optional project link"),
                            "orderLines": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "description": {"type": "string"},
                                        "count": {"type": "number"},
                                        "unitPriceExcludingVatCurrency": {"type": "number"},
                                        "vatType": _ref("VAT type: 3=25%, 31=15%, 33=12%, 5=0%, 6=0%"),
                                        "product": _ref("Optional product"),
                                    },
                                    "required": ["count", "unitPriceExcludingVatCurrency", "vatType"],
                                },
                            },
                        },
                        "required": ["customer", "orderDate", "deliveryDate", "orderLines"],
                    },
                },
            },
            "required": ["invoiceDate", "invoiceDueDate", "customer", "orders"],
        },
    },
    "create_order": {
        "method": "POST",
        "path": "/order",
        "description": "Create order (first step of invoice creation)",
        "body_schema": {
            "type": "object",
            "properties": {
                "customer": _ref("Customer"),
                "orderDate": {"type": "string", "description": "YYYY-MM-DD"},
                "deliveryDate": {"type": "string", "description": "YYYY-MM-DD"},
                "project": _ref("Optional project link"),
            },
            "required": ["customer", "orderDate", "deliveryDate"],
        },
        "needs_bank_setup": True,
    },
    "create_orderline": {
        "method": "POST",
        "path": "/order/orderline",
        "description": "Add line to order",
        "body_schema": {
            "type": "object",
            "properties": {
                "order": _ref("Order"),
                "description": {"type": "string"},
                "count": {"type": "number"},
                "unitPriceExcludingVatCurrency": {"type": "number"},
                "vatType": _ref("VAT type: 3=25%, 31=15%, 33=12%, 5=0%, 6=0%"),
                "product": _ref("Optional product reference"),
            },
            "required": ["order", "count", "unitPriceExcludingVatCurrency", "vatType"],
        },
    },
    "create_project": {
        "method": "POST",
        "path": "/project",
        "description": "Create project",
        "body_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "number": {"type": "string"},
                "projectManager": _ref("Project manager (employee)"),
                "customer": _ref("Customer"),
                "startDate": {"type": "string", "description": "YYYY-MM-DD"},
                "endDate": {"type": "string"},
            },
            "required": ["name", "projectManager", "startDate"],
        },
    },
    "create_travel_expense": {
        "method": "POST",
        "path": "/travelExpense",
        "description": "Create COMPLETE travel expense with costs and per diem. The composite tool handles cost categories, payment types, per diem rates, and delivery automatically.",
        "body_schema": {
            "type": "object",
            "properties": {
                "employee": _ref("Employee"),
                "title": {"type": "string"},
                "date": {"type": "string", "description": "YYYY-MM-DD expense date"},
                "departure_date": {"type": "string", "description": "YYYY-MM-DD trip start"},
                "return_date": {"type": "string", "description": "YYYY-MM-DD trip end"},
                "per_diem_days": {"type": "integer", "description": "Number of per diem days"},
                "per_diem_rate": {"type": "number", "description": "Daily per diem rate in NOK"},
                "costs": {
                    "type": "array",
                    "description": "Expense items (flights, taxi, hotel, etc.)",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "amount": {"type": "number", "description": "Amount in NOK"},
                            "date": {"type": "string", "description": "YYYY-MM-DD"},
                            "category": {"type": "string", "description": "e.g. Fly, Hotell, Taxi, Tog"},
                        },
                        "required": ["amount"],
                    },
                },
            },
            "required": ["employee", "title"],
        },
    },
    "create_travel_cost": {
        "method": "POST",
        "path": "/travelExpense/cost",
        "description": "Add cost to travel expense (flight, hotel, taxi, etc.)",
        "body_schema": {
            "type": "object",
            "properties": {
                "travelExpense": _ref("Travel expense report"),
                "costCategory": _ref("Cost category from get_cost_categories"),
                "amountCurrencyIncVat": {"type": "number", "description": "Amount INCLUDING VAT"},
                "date": {"type": "string", "description": "YYYY-MM-DD"},
                "paymentType": _ref("Payment type from get_travel_payment_types"),
                "comments": {"type": "string", "description": "Description of the cost"},
            },
            "required": ["travelExpense", "costCategory", "amountCurrencyIncVat", "date", "paymentType"],
        },
    },
    "create_per_diem": {
        "method": "POST",
        "path": "/travelExpense/perDiemCompensation",
        "description": "Add per diem (daily allowance / diett) to travel expense",
        "body_schema": {
            "type": "object",
            "properties": {
                "travelExpense": _ref("Travel expense report"),
                "rateCategory": _ref("Rate category from get_rate_categories (id=11 for domestic overnight)"),
                "count": {"type": "integer", "description": "Number of days"},
                "rate": {"type": "number", "description": "Daily rate in NOK (omit to use default)"},
                "location": {"type": "string", "description": "REQUIRED — city/location name"},
                "overnightAccommodation": {"type": "string", "enum": ["HOTEL", "BOARDING_HOUSE_WITHOUT_COOKING", "BOARDING_HOUSE_WITH_COOKING", "NONE"]},
            },
            "required": ["travelExpense", "rateCategory", "count", "location", "overnightAccommodation"],
        },
    },
    "create_voucher": {
        "method": "POST",
        "path": "/ledger/voucher/historical/historical",
        "description": "Create journal voucher with postings (uses historical endpoint)",
        "body_schema": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "YYYY-MM-DD"},
                "description": {"type": "string"},
                "postings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "account": _ref("Account — use account ID from get_account or create_account"),
                            "accountNumber": {"type": "integer", "description": "Account number (e.g. 1209, 6020). Always include as fallback in case account ref fails."},
                            "amountGrossCurrency": {"type": "number", "description": "Positive=debit, negative=credit. Must sum to 0."},
                            "description": {"type": "string"},
                            "customer": _ref("REQUIRED for account 1500 (Kundefordringer) — use customer ID from find_customer"),
                            "supplier": _ref("REQUIRED for account 2400 (Leverandørgjeld) — use supplier ID from find_supplier"),
                            "department": _ref("Department — use department ID from get_department. Do NOT put department in freeAccountingDimension."),
                            "project": _ref("Project — use project ID from find_project or create_project"),
                            "freeAccountingDimension1": _ref("Dimension 1 value (NOT for departments or projects)"),
                            "freeAccountingDimension2": _ref("Dimension 2 value"),
                            "freeAccountingDimension3": _ref("Dimension 3 value"),
                        },
                        "required": ["account", "amountGrossCurrency"],
                    },
                },
            },
            "required": ["date", "description", "postings"],
        },
    },
    "create_dimension_name": {
        "method": "POST",
        "path": "/ledger/accountingDimensionName",
        "description": "Create accounting dimension name",
        "body_schema": {
            "type": "object",
            "properties": {
                "dimensionName": {"type": "string"},
                "description": {"type": "string"},
                "dimensionIndex": {"type": "integer", "enum": [1, 2, 3]},
                "active": {"type": "boolean", "default": True},
            },
            "required": ["dimensionName", "dimensionIndex", "active"],
        },
    },
    "create_dimension_value": {
        "method": "POST",
        "path": "/ledger/accountingDimensionValue",
        "description": "Create accounting dimension value",
        "body_schema": {
            "type": "object",
            "properties": {
                "displayName": {"type": "string"},
                "dimensionIndex": {"type": "integer", "enum": [1, 2, 3]},
                "active": {"type": "boolean", "default": True},
                "number": {"type": "string", "description": "Unique identifier string"},
            },
            "required": ["displayName", "dimensionIndex", "active", "number"],
        },
    },
    "create_timesheet_entry": {
        "method": "POST",
        "path": "/timesheet/entry",
        "description": "Log hours on project activity",
        "body_schema": {
            "type": "object",
            "properties": {
                "employee": _ref("Employee"),
                "project": _ref("Project"),
                "activity": _ref("Activity"),
                "date": {"type": "string", "description": "YYYY-MM-DD"},
                "hours": {"type": "number"},
                "comment": {"type": "string"},
            },
            "required": ["employee", "project", "activity", "date", "hours"],
        },
    },
    "create_activity": {
        "method": "POST",
        "path": "/activity",
        "description": "Create activity for timesheet",
        "body_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "activityType": {"type": "string", "default": "GENERAL_ACTIVITY", "description": "Must be GENERAL_ACTIVITY. PROJECT_SPECIFIC_ACTIVITY cannot be created via this endpoint."},
            },
            "required": ["name", "activityType"],
        },
    },
    "link_project_activity": {
        "method": "POST",
        "path": "/project/projectActivity",
        "description": "Create and link a project-specific activity to a project. Pass activity as {name: 'Activity Name', activityType: 'PROJECT_SPECIFIC_ACTIVITY'} to create inline.",
        "body_schema": {
            "type": "object",
            "properties": {
                "activity": {
                    "type": "object",
                    "description": "Activity to link. Use {name: 'Name', activityType: 'PROJECT_SPECIFIC_ACTIVITY'} to create inline, or {id: N} for existing.",
                    "properties": {
                        "id": {"oneOf": [{"type": "integer"}, {"type": "string"}]},
                        "name": {"type": "string"},
                        "activityType": {"type": "string", "default": "PROJECT_SPECIFIC_ACTIVITY"},
                    },
                },
                "project": _ref("Project"),
            },
            "required": ["activity", "project"],
        },
    },
    "set_hourly_rate": {
        "method": "POST",
        "path": "/employee/hourlyCostAndRate",
        "description": "Set employee hourly rate",
        "body_schema": {
            "type": "object",
            "properties": {
                "employee": _ref("Employee"),
                "date": {"type": "string", "description": "YYYY-MM-DD"},
                "rate": {"type": "number", "description": "Hourly rate in NOK"},
            },
            "required": ["employee", "date", "rate"],
        },
    },
    "create_employment": {
        "method": "POST",
        "path": "/employee/employment",
        "description": "Create employment record",
        "body_schema": {
            "type": "object",
            "properties": {
                "employee": _ref("Employee"),
                "startDate": {"type": "string", "description": "YYYY-MM-DD"},
            },
            "required": ["employee", "startDate"],
        },
    },
    "create_employment_details": {
        "method": "POST",
        "path": "/employee/employment/details",
        "description": "Create employment details (percentage, salary, type, hours)",
        "body_schema": {
            "type": "object",
            "properties": {
                "employment": _ref("Employment"),
                "date": {"type": "string", "description": "Start date YYYY-MM-DD"},
                "percentageOfFullTimeEquivalent": {"type": "number", "description": "Position percentage, e.g. 80.0 for 80%"},
                "annualSalary": {"type": "number", "description": "Annual salary in NOK"},
                "hoursPerDay": {"type": "number", "description": "Standard working hours per day, e.g. 7.5"},
            },
            "required": ["employment", "date"],
        },
    },
    "create_salary_transaction": {
        "method": "POST",
        "path": "/salary/transaction",
        "description": "Create salary/payroll transaction",
        "body_schema": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "YYYY-MM-DD"},
                "year": {"type": "integer"},
                "month": {"type": "integer"},
                "payslips": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "employee": _ref("Employee"),
                            "date": {"type": "string"},
                            "year": {"type": "integer"},
                            "month": {"type": "integer"},
                            "specifications": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "salaryType": _ref("Salary type from get_salary_types"),
                                        "rate": {"type": "number"},
                                        "count": {"type": "number", "default": 1},
                                        "description": {"type": "string"},
                                    },
                                    "required": ["salaryType", "rate", "count"],
                                },
                            },
                        },
                        "required": ["employee", "date", "year", "month", "specifications"],
                    },
                },
            },
            "required": ["date", "year", "month", "payslips"],
        },
    },
    "create_incoming_invoice": {
        "method": "POST",
        "path": "/incomingInvoice",
        "description": "Register incoming/supplier invoice",
        "body_schema": {
            "type": "object",
            "properties": {
                "invoiceHeader": {
                    "type": "object",
                    "properties": {
                        "vendorId": _id_or_ref("Supplier ID"),
                        "invoiceDate": {"type": "string", "description": "YYYY-MM-DD"},
                        "dueDate": {"type": "string", "description": "YYYY-MM-DD"},
                        "invoiceAmount": {"type": "number", "description": "Total amount incl. VAT"},
                        "description": {"type": "string"},
                        "invoiceNumber": {"type": "string"},
                    },
                    "required": ["vendorId", "invoiceDate", "dueDate", "invoiceAmount"],
                },
                "orderLines": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "row": {"type": "integer", "default": 0},
                            "description": {"type": "string"},
                            "accountId": _id_or_ref("Expense account ID from get_account"),
                            "amountInclVat": {"type": "number"},
                            "vatTypeId": _id_or_ref("Inbound VAT: 1=25%, 11=15%, 12=12%"),
                        },
                        "required": ["accountId", "amountInclVat", "vatTypeId"],
                    },
                },
            },
            "required": ["invoiceHeader", "orderLines"],
        },
    },

    # ── Actions (PUT) ─────────────────────────────────────
    "convert_order_to_invoice": {
        "method": "PUT",
        "path": "/order/{order_id}/:invoice",
        "description": "Convert order to invoice",
        "path_params": ["order_id"],
        "params_schema": {
            "type": "object",
            "properties": {
                "invoiceDate": {"type": "string", "description": "YYYY-MM-DD"},
            },
            "required": ["invoiceDate"],
        },
        "body_schema": None,
    },
    "register_payment": {
        "method": "PUT",
        "path": "/invoice/{invoice_id}/:payment",
        "description": "Register payment on invoice",
        "path_params": ["invoice_id"],
        "params_schema": {
            "type": "object",
            "properties": {
                "paymentDate": {"type": "string", "description": "YYYY-MM-DD"},
                "paymentTypeId": _id_or_ref("Payment type ID from get_payment_types"),
                "paidAmount": {"type": "number"},
                "paidAmountCurrency": {"type": "number"},
            },
            "required": ["paymentDate", "paymentTypeId", "paidAmount", "paidAmountCurrency"],
        },
        "body_schema": None,
    },
    "create_credit_note": {
        "method": "PUT",
        "path": "/invoice/{invoice_id}/:createCreditNote",
        "description": "Create credit note for invoice",
        "path_params": ["invoice_id"],
        "params_schema": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "YYYY-MM-DD"},
            },
            "required": ["date"],
        },
        "body_schema": None,
    },
    "deliver_travel_expense": {
        "method": "PUT",
        "path": "/travelExpense/:deliver",
        "description": "Submit/deliver travel expense report",
        "body_schema": {
            "type": "array",
            "items": _ref("Travel expense to deliver"),
            "description": "Array of {id: N} references",
        },
    },
    "reverse_voucher": {
        "method": "PUT",
        "path": "/ledger/voucher/{voucher_id}/:reverse",
        "description": "Reverse a voucher (e.g. to undo a payment)",
        "path_params": ["voucher_id"],
        "params_schema": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "YYYY-MM-DD"},
            },
            "required": ["date"],
        },
        "body_schema": None,
    },
    "update_employee": {
        "method": "PUT",
        "path": "/employee/{employee_id}",
        "description": "Update employee (must include id + version)",
        "path_params": ["employee_id"],
        "body_schema": {
            "type": "object",
            "properties": {
                "id": _id_or_ref("Employee ID"),
                "version": _id_or_ref("Version from GET response"),
                "firstName": {"type": "string"},
                "lastName": {"type": "string"},
                "email": {"type": "string"},
                "phoneNumberMobile": {"type": "string"},
                "dateOfBirth": {"type": "string"},
                "employeeNumber": {"type": "string"},
                "department": _ref("Department"),
                "userType": {"type": "string", "enum": ["STANDARD", "EXTENDED", "NO_ACCESS"]},
            },
            "required": ["id", "version"],
        },
    },
    "update_customer": {
        "method": "PUT",
        "path": "/customer/{customer_id}",
        "description": "Update customer (must include id + version)",
        "path_params": ["customer_id"],
        "body_schema": {
            "type": "object",
            "properties": {
                "id": _id_or_ref("Customer ID"),
                "version": _id_or_ref("Version from GET response"),
                "name": {"type": "string"},
                "email": {"type": "string"},
                "phoneNumber": {"type": "string"},
                "invoiceEmail": {"type": "string"},
                "organizationNumber": {"type": "string"},
            },
            "required": ["id", "version"],
        },
    },
    "update_project": {
        "method": "PUT",
        "path": "/project/{project_id}",
        "description": "Update project (e.g. set fixed price)",
        "path_params": ["project_id"],
        "body_schema": {
            "type": "object",
            "properties": {
                "id": _id_or_ref("Project ID"),
                "version": _id_or_ref("Version from GET response"),
                "name": {"type": "string"},
                "projectManager": _ref("Project manager"),
                "startDate": {"type": "string"},
                "isFixedPrice": {"type": "boolean"},
                "fixedprice": {"type": "number"},
            },
            "required": ["id", "version", "name", "projectManager", "startDate"],
        },
    },
    "setup_bank_account": {
        "method": "PUT",
        "path": "/ledger/account/{account_id}",
        "description": "Set bank account number on account 1920 (required for invoicing)",
        "path_params": ["account_id"],
        "body_schema": {
            "type": "object",
            "properties": {
                "id": _id_or_ref("Account ID from get_account"),
                "version": _id_or_ref("Version from get_account"),
                "number": {"type": "integer", "default": 1920},
                "name": {"type": "string", "default": "Bankinnskudd"},
                "isBankAccount": {"type": "boolean", "default": True},
                "bankAccountNumber": {"type": "string", "default": "15045251362"},
            },
            "required": ["id", "version", "number", "name", "isBankAccount", "bankAccountNumber"],
        },
    },
}

# Common hallucinated aliases
OPERATIONS["find_department"] = OPERATIONS["get_department"]

# All valid operation IDs
OPERATION_IDS = sorted(OPERATIONS.keys())
