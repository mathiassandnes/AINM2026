# Trace Tracker — Challenge 1

Each non-perfect run is a ticket. Status: FIXED (deployed), OPEN (needs fix), WONTFIX.

## Revision 00053-pgq (latest)
Fixes: find-then-create skip bug (continue→break), ledger error correction → fallback.

## Revision 00052-v58

| Trace ID | Task | Score | Status | Issue |
|----------|------|-------|--------|-------|
| d5ef158d | Create supplier (PT) | plan_success | OK | - |
| 55141a1a | Credit note (NO) | plan_success | OK | - |
| 82584420 | Reverse payment (NO) | plan_success | OK | - |
| 495d0e72 | Travel expense (NO) | plan_success | OK | - |
| e94d85b5 | Project fixed price (NO) | plan_success | OK | - |
| f2e3ae8e | Ledger error correction (NO) | fallback 2/11 | FIXED-00053 | find_match on voucher description for account numbers. Now goes to fallback directly. |
| 8224f6de | Full project lifecycle (PT) | fallback 4/20 | FIXED-00053 | find-then-create skip broken (continue vs break bug) |

## Revision 00051-2lk

| Trace ID | Task | Score | Status | Issue |
|----------|------|-------|--------|-------|
| b1672bf5 | Reminder fee (NN) | plan_success 8/8 | OK | - |
| 3514ba97 | Credit note (EN) | plan_success 3/3 | OK | - |
| 0b04d93d | Project fixed price (NO) | plan_success 5/5 | OK | - |
| 0cee8e38 | Create employee (DE) | plan_success 3/3 | OK | - |
| add9dffa | Reminder fee (EN) | plan_success 7/7 | OK | - |
| 81fecf35 | Reminder fee voucher (DE) | fallback 3/7 | FIXED-00052 | customer: {} on acct 1500 — auto-inject didn't find nested customer |
| a6479a12 | Bank reconciliation (DE) | fallback 11/15 | FIXED-00052 | dateFrom == dateTo on voucher search |
| e7322cfe | Employee onboard (NN) | fallback 3/6 | FIXED-00052 | department: {} not stripped → validation error |
| d7a03d1b | Ledger analysis + projects (NO) | fallback 3/10 | OPEN | Activity linking: GENERAL_ACTIVITY can't link |

## Revision 00050-qz4
Fixes: find-then-create skip, activity linking guidance, path param validation, no hardcoded IDs.

## Revision 00046-8pd

| Trace ID | Task | Score | Status | Issue |
|----------|------|-------|--------|-------|
| 2493dc77 | Ledger analysis + projects (NN) | 0/10 | FIXED-00050 | projectManager: {} (fake employee params). No hardcoded IDs rule added. |
| 79e62bdc | Overdue invoice + reminder (EN) | 7/10 | FIXED-00050 | paymentTypeId: "1" string + unresolved path param |
| 62b909d4 | Employee from contract (NN) | 0/10 | FIXED-00050 | Dept not found → empty ref. find-then-create skip added. |
| 2493dc77 | Ledger analysis + projects (NN) | 0/10 | FIXED-00050 | Activity linking: now uses PROJECT_GENERAL_ACTIVITY. |

## Revision 00045-fmd

| Trace ID | Task | Score | Status | Issue |
|----------|------|-------|--------|-------|
| 29b9651e | Full project lifecycle (DE) | 0/10 | OPEN-3 | find-then-create not conditional: created duplicate customer+employee. Activity-project linking broken. |
| 0b2ecc77 | Order→invoice→payment (DE) | 7/10 | OPEN-3 | find-then-create: found product then tried to create duplicate. Fallback completed. |
| 53a4ffda | Employee from contract (EN) | 0/10 | OPEN-4 | Wrong entity type (find_customer for department). Wrong $step ref. |

## Revision 00044-gsw

| Trace ID | Task | Score | Status | Issue |
|----------|------|-------|--------|-------|
| 6920e9b0 | Project fixed price (EN) | 8/8 100% | OK | - |
| c7f66d84 | Order→invoice→payment (ES) | 8/8 100% | OK | - |
| 6e7c8141 | Ledger error correction (EN) | 13/22 59% | OK | Partial — complex multi-step |
| 063b0e37 | FX disagio voucher (NO) | 2/11 18% | FIXED-00046 | Voucher acct 1500 missing customer_id |
| f9522646 | Receipt/expense posting (ES) | 0/10 | OPEN-5 | Wrong account numbers (6360, 7340, 1910 don't exist). Fallback death spiral paginating accounts. |
| 6bae1eea | Ledger analysis + projects (PT) | 0/10 | OPEN-6 | Hardcoded projectManager ID "1". Guessed accounts without querying ledger. |
| 7c237a8b | Month-end closing (NN) | fallback | FIXED-00045 | Account 6030 doesn't exist |
| 648ecd5d | Year-end closing (DE) | fallback 13/14 | FIXED-00045 | Account 8700 empty ref |

## Open Issues

### OPEN-3: Find-then-create not conditional
- **Traces:** 29b9651e, 0b2ecc77, 53a4ffda
- **Root cause:** Planner generates both find_X and create_X steps. Executor runs create even when find succeeded, causing duplicates and 422 errors.
- **Fix:** Executor should skip create steps when the preceding find step returned results. OR planner should not include redundant create steps.

### OPEN-4: Wrong entity type / wrong $step_N ref
- **Traces:** 53a4ffda
- **Root cause:** Planner used find_customer to search for a department. Referenced wrong step.
- **Fix:** Planner rules need to be clearer. Partially addressed in 00045 with better rules.

### OPEN-5: Fallback account discovery death spiral
- **Traces:** f9522646
- **Root cause:** Haiku wastes 15 turns paginating /ledger/account. Name filter doesn't work (returns all). No number range filter available.
- **Fix:** Add common account number cheat sheet to fallback prompt (done in 00045). Consider adding a search_account composite tool.

### OPEN-6: Hardcoded IDs and guessed data
- **Traces:** 6bae1eea
- **Root cause:** Planner hardcodes projectManager ID as "1" and guesses account names without querying ledger.
- **Fix:** Planner rule: NEVER hardcode IDs. Always use find_employee for project managers.
