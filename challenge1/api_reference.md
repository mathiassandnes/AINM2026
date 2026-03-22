# Tripletex API Reference — Extracted from OpenAPI Spec v2.74.00

Source: `https://kkpqfuj-amager.tripletex.dev/v2/openapi.json` (3.6MB)
Extracted: 2026-03-20

This file contains the **exact** schema definitions for endpoints our agent uses.
Use this as the source of truth when building/fixing composite tools.

---

## TravelExpense (POST /travelExpense)

The body is a `TravelExpense` object.

### TravelExpense schema
| Field | Type | RO | Notes |
|-------|------|----|-------|
| id | integer | | |
| version | integer | | |
| employee | Employee | | `{"id": N}` |
| title | string | | |
| project | Project | | `{"id": N}` |
| department | Department | | `{"id": N}` |
| **travelDetails** | **TravelDetails** | | **Nested object — departure/return dates go HERE** |
| isChargeable | boolean | | |
| travelAdvance | number | | |
| costs | array of Cost | | Inline costs |
| perDiemCompensations | array of PerDiemCompensation | | Inline per diem |
| date | string | YES | Readonly — derived from costs/per diem |
| amount | number | YES | |
| state | string | YES | enum: ALL, REJECTED, OPEN, APPROVED, SALARY_PAID, DELIVERED |
| voucher | Voucher | | |
| invoice | Invoice | | |
| payslip | Payslip | | |
| vatType | VatType | | |
| paymentCurrency | Currency | | |
| attachment | Document | | |
| attestationSteps | array | | |
| attestation | Attestation | | |
| freeDimension1 | AccountingDimensionValue | | |
| freeDimension2 | AccountingDimensionValue | | |
| freeDimension3 | AccountingDimensionValue | | |
| isFixedInvoicedAmount | boolean | | |
| isMarkupInvoicedPercent | boolean | | |
| isIncludeAttachedReceiptsWhenReinvoicing | boolean | | |
| fixedInvoicedAmount | number | | |
| markupInvoicedPercent | number | | |

### TravelDetails schema (nested inside TravelExpense)
| Field | Type | Notes |
|-------|------|-------|
| isForeignTravel | boolean | |
| isDayTrip | boolean | |
| isCompensationFromRates | boolean | |
| **departureDate** | **string** | **YYYY-MM-DD** |
| **returnDate** | **string** | **YYYY-MM-DD** |
| detailedJourneyDescription | string | |
| departureFrom | string | |
| destination | string | |
| departureTime | string | |
| returnTime | string | |
| purpose | string | |

**CRITICAL**: `departureDate` and `returnDate` are NOT top-level fields on TravelExpense.
They must be nested: `{"travelDetails": {"departureDate": "2026-01-20", "returnDate": "2026-01-24"}}`

---

## Cost (POST /travelExpense/cost)

| Field | Type | Notes |
|-------|------|-------|
| travelExpense | TravelExpense | `{"id": N}` — required |
| costCategory | TravelCostCategory | `{"id": N}` |
| paymentType | TravelPaymentType | `{"id": N}` |
| vatType | VatType | `{"id": N}` |
| currency | Currency | `{"id": N}` |
| category | string | |
| comments | string | |
| rate | number | |
| amountCurrencyIncVat | number | Amount including VAT |
| amountNOKInclVAT | number | |
| date | string | YYYY-MM-DD |
| isChargeable | boolean | |
| participants | array of CostParticipant | |

---

## PerDiemCompensation (POST /travelExpense/perDiemCompensation)

| Field | Type | Notes |
|-------|------|-------|
| travelExpense | TravelExpense | `{"id": N}` — required |
| rateType | TravelExpenseRate | `{"id": N}` |
| rateCategory | TravelExpenseRateCategory | `{"id": N}` — use `/travelExpense/rateCategory` to find |
| countryCode | string | e.g. "NO" |
| travelExpenseZoneId | integer | Optional, falls back to zone field |
| **overnightAccommodation** | **string** | **enum: NONE, HOTEL, BOARDING_HOUSE_WITHOUT_COOKING, BOARDING_HOUSE_WITH_COOKING** |
| location | string | |
| address | string | |
| **count** | **integer** | **Number of days (NOT "countDays")** |
| rate | number | |
| amount | number | |
| isDeductionForBreakfast | boolean | |
| isDeductionForLunch | boolean | |
| isDeductionForDinner | boolean | |

**IMPORTANT**: The field is `count`, not `countDays`. The travel expense must have
`travelDetails.departureDate` and `travelDetails.returnDate` set BEFORE adding per diem.

---

## IncomingInvoice (POST /incomingInvoice)

Body is `IncomingInvoiceAggregateExternalWrite`:

### IncomingInvoiceAggregateExternalWrite
| Field | Type | Notes |
|-------|------|-------|
| version | integer | Voucher version |
| invoiceHeader | IncomingInvoiceHeaderExternalWrite | |
| orderLines | array of IncomingOrderLineExternalWrite | |

### IncomingInvoiceHeaderExternalWrite
| Field | Type | Notes |
|-------|------|-------|
| vendorId | integer | Supplier ID |
| invoiceDate | string | YYYY-MM-DD |
| dueDate | string | YYYY-MM-DD |
| currencyId | integer | Optional |
| invoiceAmount | number | Amount including VAT |
| description | string | |
| invoiceNumber | string | |
| voucherTypeId | integer | |
| purchaseOrderId | integer | |

### IncomingOrderLineExternalWrite
| Field | Type | Notes |
|-------|------|-------|
| externalId | string | Unique ID for validation |
| **row** | **integer** | **Starts at 1 (NOT 0)** |
| description | string | |
| accountId | integer | Account ID (NOT number — must resolve first) |
| count | number | Max 10 decimals |
| amountInclVat | number | Max 2 decimals |
| vatTypeId | integer | |
| departmentId | integer | |
| projectId | integer | |
| employeeId | integer | |
| productId | integer | |
| assetId | integer | |
| customerId | integer | |
| vendorId | integer | |
| taxTransactionTypeId | integer | |
| freeDimension1Id | integer | |
| freeDimension2Id | integer | |
| freeDimension3Id | integer | |

**KNOWN ISSUE**: POST /incomingInvoice returns 403 on competition sandboxes (module not enabled).
Fallback: create a voucher with expense debit + VAT debit + AP credit.

---

## SalaryTransaction (POST /salary/transaction)

### SalaryTransaction
| Field | Type | Notes |
|-------|------|-------|
| date | string | Voucher date YYYY-MM-DD |
| year | integer | |
| month | integer | |
| isHistorical | boolean | For pre-opening-balance entries |
| paySlipsAvailableDate | string | Defaults to voucherDate |
| payslips | array of Payslip | |

### Payslip
| Field | Type | Notes |
|-------|------|-------|
| employee | Employee | `{"id": N}` |
| date | string | YYYY-MM-DD |
| year | integer | |
| month | integer | |
| specifications | array of SalarySpecification | |
| department | Department | `{"id": N}` |

### SalarySpecification
| Field | Type | Notes |
|-------|------|-------|
| rate | number | |
| count | number | |
| project | Project | `{"id": N}` |
| department | Department | `{"id": N}` |
| salaryType | SalaryType | `{"id": N}` — use `/salary/type` to find |
| employee | Employee | `{"id": N}` |
| description | string | |
| year | integer | |
| month | integer | |
| amount | number | |

---

## Voucher (POST /ledger/voucher)

### Voucher
| Field | Type | Notes |
|-------|------|-------|
| date | string | YYYY-MM-DD |
| description | string | |
| voucherType | VoucherType | `{"id": N}` |
| postings | array of Posting | |
| document | Document | |
| attachment | Document | |
| externalVoucherNumber | string | Max 70 chars |
| vendorInvoiceNumber | string | |

### Posting
| Field | Type | Notes |
|-------|------|-------|
| account | Account | `{"id": N}` — must be real account ID, not number |
| description | string | |
| amountGrossCurrency | number | Positive = debit, negative = credit |
| amount | number | |
| amountCurrency | number | |
| amountGross | number | |
| vatType | VatType | `{"id": N}` — **REQUIRED for VAT postings** |
| customer | Customer | `{"id": N}` |
| supplier | Supplier | `{"id": N}` |
| employee | Employee | `{"id": N}` |
| project | Project | `{"id": N}` |
| product | Product | `{"id": N}` |
| department | Department | `{"id": N}` |
| currency | Currency | `{"id": N}` |
| date | string | |
| invoiceNumber | string | |
| row | integer | |
| freeAccountingDimension1 | AccountingDimensionValue | |
| freeAccountingDimension2 | AccountingDimensionValue | |
| freeAccountingDimension3 | AccountingDimensionValue | |

**CRITICAL for voucher postings**: When posting to a VAT account (e.g. 2710 inbound MVA),
you likely need to set `vatType` on the expense posting for Tripletex to handle VAT correctly.
The validation error "Et bilag kan..." often means VAT type is missing on postings.

---

## Bugs Found From This Spec

1. **TravelExpense**: `departureDate`/`returnDate` must be in `travelDetails` sub-object, NOT top-level
2. **PerDiemCompensation**: Field is `count` not `countDays`
3. **IncomingInvoice orderLines**: `row` starts at **1**, not 0
4. **Voucher postings**: May need `vatType` on expense postings for VAT validation
5. **TravelExpense.date**: Is **readonly** — cannot be set on POST, it's derived
