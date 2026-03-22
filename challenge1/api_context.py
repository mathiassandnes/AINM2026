"""
Compact Tripletex API reference.
Only endpoints and fields — behavioral guidance is in the system prompt.
Composite tool details are in tool schemas.
Full body examples only for endpoints that need api_post/api_put (dimensions, timesheet, etc.).
"""


def get_system_prompt() -> str:
    return """## Tripletex API Reference
- List responses: {"fullResultSize": N, "values": [...]}
- Single resource: {"value": {...}}
- Errors: {"error": true, "status_code": N, "validationMessages": [...]}
- Fields filter: {"fields": "id,name,..."}. Use "*" to see all fields.
- Dates: YYYY-MM-DD. References: {"id": N}.

### Employee
- GET /employee — params: firstName, lastName, email, employeeNumber
- POST /employee — REQUIRED: firstName, lastName, userType("STANDARD"), department({"id": N})
- PUT /employee/{id} — update (include version)
- POST /employee/employment — {"employee": {"id": N}, "startDate": "YYYY-MM-DD"}
- POST /employee/employment/details — percentage, annualSalary, hoursPerWeek

### Customer
- GET /customer — params: name, email, organizationNumber
- POST /customer — REQUIRED: name. Optional: email, organizationNumber, phoneNumber, invoiceEmail, postalAddress

### Supplier
- GET /supplier — params: name
- POST /supplier — {"name": "...", "isSupplier": true, "email": "...", "organizationNumber": "..."}

### Product
- GET /product — params: name, number
- POST /product — REQUIRED: name. Optional: number, priceExcludingVatCurrency, vatType({"id": N})

### Invoice
- GET /invoice — REQUIRES: invoiceDateFrom, invoiceDateTo. Optional: customerId
  Valid fields: id, invoiceNumber, invoiceDate, invoiceDueDate, amount, amountCurrency, amountOutstanding, customer, voucher
  NOTE: Use ONLY these field names. Do NOT add Currency suffix (amountOutstandingCurrency does NOT exist).
- POST /invoice — create from orders: {"invoiceDate", "invoiceDueDate", "customer": {"id": N}, "orders": [{"id": N}]}
- PUT /invoice/{id}/:payment — QUERY PARAMS (not body): paymentDate, paymentTypeId, paidAmount, paidAmountCurrency
- PUT /invoice/{id}/:send — params: sendType ("EMAIL")
- PUT /invoice/{id}/:createCreditNote — params: date

### Order
- POST /order — {"customer": {"id": N}, "deliveryDate", "orderDate", "project": {"id": N}}
- POST /order/orderline — {"order": {"id": N}, "description", "count", "unitPriceExcludingVatCurrency", "vatType": {"id": N}}
- PUT /order/{id}/:invoice — params: invoiceDate (converts order to invoice)

### Project
- POST /project — REQUIRED: name, projectManager({"id": EMPLOYEE_ID}), startDate. Optional: customer, number
- PUT /project/{id} — isFixedPrice, fixedprice (include version)
- GET /project — params: name

### Department
- GET /department — params: name
- POST /department — {"name": "...", "departmentNumber": "..."}

### Travel Expense
- POST /travelExpense — {"employee": {"id": N}, "title": "...", "travelDetails": {"departureDate", "returnDate", "isCompensationFromRates": true}}
- POST /travelExpense/cost — {"travelExpense": {"id": N}, "costCategory": {"id": N}, "amountCurrencyIncVat": 500.0, "date", "paymentType": {"id": N}, "comments": "..."}
- POST /travelExpense/perDiemCompensation — {"travelExpense": {"id": N}, "rateCategory": {"id": N}, "count": DAYS, "location": "...", "overnightAccommodation": "HOTEL"|"NONE"}
- PUT /travelExpense/:deliver — body: [{"id": N}]
- GET /travelExpense/costCategory — list cost categories
- GET /travelExpense/rateCategory — list per diem rate categories

### Voucher / Ledger
- POST /ledger/voucher — {"date", "description", "postings": [{"account": {"id": N}, "amountGrossCurrency": 100.0, "description": "...", "freeAccountingDimension1": {"id": N}}]}
  Positive amount = debit, negative = credit. Postings must balance.
- GET /ledger/voucher — REQUIRES: dateFrom, dateTo
- PUT /ledger/voucher/{id}/:reverse — params: date
- GET /ledger/account — params: number
- POST /ledger/account — {"number": 6030, "name": "Avskrivning programvare"}
- GET /ledger/posting — params: accountId, accountNumberFrom, accountNumberTo, dateFrom, dateTo
  fields: id,date,description,amount,amountCurrency,account(id,number,name),voucher(id,number,description)

### Free Accounting Dimensions
- POST /ledger/accountingDimensionName — {"dimensionName": "Region", "dimensionIndex": 1, "active": true}
  dimensionIndex: 1, 2, or 3 (maps to freeAccountingDimension1/2/3 on voucher postings)
- POST /ledger/accountingDimensionValue — {"displayName": "Vestlandet", "dimensionIndex": 1, "active": true, "number": "1"}
- GET /ledger/accountingDimensionValue/search — params: dimensionIndex
Pattern: create dimension name → create values → use in create_voucher with dimension1_id

### Timesheet
- POST /timesheet/entry — {"employee": {"id": N}, "project": {"id": N}, "activity": {"id": N}, "date", "hours": 8.0}
- POST /project/projectActivity — {"project": {"id": N}, "activity": {"name": "...", "activityType": "PROJECT_SPECIFIC_ACTIVITY"}}
- POST /employee/hourlyCostAndRate — {"employee": {"id": N}, "date", "rate": 1450.0}

### Incoming Invoice
- POST /incomingInvoice — {"invoiceHeader": {"vendorId": SUPPLIER_ID, "invoiceDate", "dueDate", "invoiceAmount": INCL_VAT, "invoiceNumber": "..."}, "orderLines": [{"accountId": EXPENSE_ACCT_ID, "amountInclVat", "vatTypeId": 1, "description": "..."}]}
  vatTypeId for incoming: 1=25%, 11=15%, 12=12%

### Salary
- POST /salary/transaction — {"date", "year", "month", "payslips": [{"employee": {"id": N}, "date", "year", "month", "specifications": [{"salaryType": {"id": N}, "rate": 49550.0, "count": 1}]}]}
- GET /salary/type — find salary type IDs (number "2000"=Fastlønn, "2002"=Bonus)

### Payment Types
- GET /invoice/paymentType — list payment types for invoices
"""
