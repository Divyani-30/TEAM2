# HELIX GLOBAL CORP - HR INTELLIGENCE BOT
## RAG-Based System Documentation & Data Analysis Report

**Project:** HR Intelligence Bot (Hackathon Task 2)  
**Organization:** Helix Global Corp  
**Date:** February 5, 2026  
**Document Version:** 1.0

---

## EXECUTIVE SUMMARY

This document provides comprehensive documentation for a Retrieval Augmented Generation (RAG) based HR Intelligence Bot system designed for Helix Global Corp. The system integrates multiple data sources to answer employee queries about leave policies, attendance compliance, tenure benefits, and regional HR regulations.

### Key Components:
- **4 primary data sources** (CSV, JSON, XLSX, PDF)
- **500+ employee records** across 8 global locations
- **Multi-sheet analytics** with leave balances and history
- **Detailed attendance tracking** with biometric data
- **Complex policy framework** with regional variations

---

## 1. SYSTEM ARCHITECTURE

### 1.1 RAG System Overview

The HR Intelligence Bot follows a RAG architecture pattern:

```
User Query → Document Retrieval → Context Extraction → LLM Processing → Grounded Response
```

**Key Principles:**
- All responses must be grounded in the provided data sources
- No hallucinations or invented information
- Cross-reference between data sources for accuracy
- Policy compliance validation against HR manual

### 1.2 Data Source Hierarchy

**Priority Order for Query Resolution:**
1. **Policy Manual (PDF)** - Authoritative source for rules and regulations
2. **Employee Master (CSV)** - Ground truth for employee demographics and tenure
3. **Leave Intelligence (XLSX)** - Real-time leave balances and history
4. **Attendance Logs (JSON)** - Detailed biometric and time-tracking data

---

## 2. DATA SOURCE DOCUMENTATION

### 2.1 Employee Master Database
**File:** `employee_master.csv`  
**Size:** 62 KB  
**Records:** 500+ employees  
**Primary Key:** `emp_id`

#### Schema Definition:

| Field Name | Data Type | Description | Sample Value |
|------------|-----------|-------------|--------------|
| `emp_id` | String | Unique employee identifier (PRIMARY KEY) | EMP1001 |
| `name` | String | Full employee name | Patrick Sanchez |
| `dept` | String | Department assignment | Engineering, Marketing, Finance, HR, Legal, Product, IT, Operations, Customer Success |
| `location` | String | Office location | Singapore, London, Bangalore, Tokyo, Sydney, New York, Berlin |
| `role` | String | Job title/position | Corporate treasurer, Risk analyst, etc. |
| `joining_date` | Date (YYYY-MM-DD) | Date of employment start | 2023-07-14 |
| `salary_band` | String (A-E) | Compensation tier | A, B, C, D, E |
| `email` | String | Corporate email address | rhodespatricia@example.org |
| `manager_id` | String | Reporting manager's emp_id | EMP1029 (nullable) |
| `is_active` | Boolean | Current employment status | True/False |
| `performance_rating` | String | Latest performance review | Outstanding, Exceeds, Meets, Needs Improvement |
| `certifications` | String | Professional certifications | AWS, CISSP, PMP, CPA, SCRUM, None |

#### Data Quality Notes:
- **Manager ID:** Some employees (senior leadership) have null manager_id values
- **Active Status:** Includes both active (True) and inactive (False) employees
- **Joining Dates:** Range from 2018 to 2026 (future dates may indicate planned hires)
- **Global Distribution:** 8 distinct office locations

#### Use Cases:
- Tenure calculation for loyalty benefits
- Department and location-based policy application
- Manager hierarchy for leave approvals
- Active employee filtering

---

### 2.2 Attendance Logs (Detailed Biometric Data)
**File:** `attendance_logs_detailed.json`  
**Size:** 12 MB  
**Format:** Nested JSON structure  
**Primary Key:** `emp_id` (top-level keys)

#### JSON Structure:

```json
{
  "EMP1001": {
    "period": "Nov-Jan 2025-2026",
    "total_days": 65,
    "records": [
      {
        "date": "2025-11-03",
        "check_in": "08:49",
        "check_out": null,
        "location_logged": "Office",
        "metadata": {
          "device": "Mobile-App",
          "ip": "160.213.53.81",
          "login_attempts": 3
        }
      }
    ]
  }
}
```

#### Schema Definition:

**Top Level:**
- `emp_id` (key): Employee identifier
- `period`: Reporting period string
- `total_days`: Number of days in the record set
- `records`: Array of daily attendance entries

**Records Array:**
- `date` (String, YYYY-MM-DD): Attendance date
- `check_in` (String, HH:MM): Entry time (24-hour format)
- `check_out` (String/null, HH:MM): Exit time or null if missing
- `location_logged` (String): "Office", "Remote", "WFH"
- `metadata` (Object): System tracking information
  - `device`: Device type (Mobile-App, Desktop, Biometric)
  - `ip`: IP address for login
  - `login_attempts`: Authentication attempts count

#### Critical Data Points:
- **Missing Check-outs:** Indicated by `null` values in `check_out` field
- **Policy Implication:** >5 missing check-outs/month = 2% salary deduction
- **Metadata:** System information for audit purposes (not typically used in policy calculations)

#### Use Cases:
- Attendance compliance monitoring
- Missing check-out violation tracking
- Work location tracking (office vs. remote)
- Audit trail for disputes

---

### 2.3 Leave Intelligence Workbook
**File:** `leave_intelligence.xlsx`  
**Size:** 88 KB  
**Sheets:** 4 (3 active, 1 archived)

#### Sheet 1: Leave_History
**Purpose:** Historical record of all leave applications

| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| `emp_id` | String | Employee identifier |
| `leave_type` | String | Annual, Sick, Loyalty, Emergency, Maternity, Paternity |
| `days` | Integer | Number of days requested |
| `start_date` | Date | Leave start date |
| `status` | String | Approved, Rejected, Pending |
| `approver` | String | Manager emp_id who processed request |

**Sample Data:**
```
emp_id    leave_type  days  start_date   status    approver
EMP1001   Sick        5     2025-04-23   Approved  EMP1339
EMP1002   Loyalty     1     2025-06-05   Rejected  EMP1039
EMP1003   Emergency   4     2026-01-23   Rejected  EMP1097
```

**Use Cases:**
- Leave pattern analysis
- Approval rate tracking
- Historical trend analysis
- Manager decision patterns

---

#### Sheet 2: Available_Balances
**Purpose:** Current leave balances for all employees (CRITICAL for eligibility checks)

| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| `emp_id` | String | Employee identifier (PRIMARY KEY) |
| `annual_bal` | Integer | Remaining annual leave days |
| `sick_bal` | Integer | Remaining sick leave days |
| `loyalty_bal` | Integer | Remaining loyalty bonus days |
| `emergency_bal` | Integer | Remaining emergency leave days |

**Sample Data:**
```
emp_id    annual_bal  sick_bal  loyalty_bal  emergency_bal
EMP1001   8           8         1            2
EMP1002   9           7         1            0
EMP1003   7           9         2            3
EMP1004   14          10        2            0
```

**Policy Validation Rules:**
- Annual leave requests require `annual_bal >= requested_days`
- Loyalty balance only appears for employees with 3+ years tenure
- Emergency balance caps at 3 days/year
- Sick balance caps at 10 days/year

---

#### Sheet 3: Dept_Analytics
**Purpose:** Departmental aggregations and analytics
**Note:** Structure not fully documented; use with caution

---

#### Sheet 4: ARCHIVED_2024
**Purpose:** Historical data from previous year
**Status:** ⚠️ **DO NOT USE - Marked as ARCHIVED per Readme.txt**

---

### 2.4 HR Policy Manual (Authoritative Rulebook)
**File:** `Helix_Pro_Policy_v2.pdf`  
**Version:** 2026.01  
**Effective Date:** January 1, 2026  
**Pages:** 13  
**Classification:** Confidential & Proprietary

#### Policy Structure:

**Section 1: Introduction & Scope**
- Applies to all permanent employees globally
- Managers responsible for consistent implementation

**Section 2: Leave Categories & Entitlements**

| Leave Type | Days/Year | Advance Notice | Documentation |
|------------|-----------|----------------|---------------|
| Annual Leave | 15 days | 2 weeks | Manager approval |
| Sick Leave | 10 days | N/A | MC required >2 consecutive days |
| Emergency Leave | 3 days | N/A | Manager discretion + documentation |

**Carryover Policy:** Annual leave cannot be carried beyond March 31 of following year

---

**Section 3: Tenure-Based Loyalty Benefits**

Critical calculation: `tenure = current_date - joining_date`

| Tier | Service Period | Benefit | Total Annual Leave |
|------|----------------|---------|-------------------|
| TIER 1 | 3-5 years | +2 days | 17 days |
| TIER 2 | 5+ years | +5 days | 20 days |
| Sabbatical | 7+ years | 15 days (separate) | Available annually |

**Important Notes:**
- Eligibility begins on anniversary date (exact 3 or 5 years)
- Breaks in service reset the tenure counter
- Tenure calculated from `joining_date` field in employee master

---

**Section 4: Regional Policy Variations**

Regional exceptions that override global policies:

**BANGALORE, INDIA:**
- +1 day Diwali bonus leave
- Festival advance payment in October

**NEW YORK, USA:**
- 401k matching from day one
- Standard US federal holidays

**TOKYO, JAPAN:**
- Golden Week provisions: 5 consecutive days
- Cultural holiday observances

---

**Section 5: Singapore-Specific Requirements** ⚠️ **CRITICAL COMPLIANCE**

**MANDATORY RULE:**
Employees in Singapore (`location = 'Singapore'`) MUST provide a valid medical certificate (MC) for **ALL** sick leave applications, regardless of duration.

Includes:
- Single-day sick leave ✓
- Half-day sick leave ✓
- Extended medical leave ✓

**Non-Compliance:** Leave classified as unpaid absence

**MC Requirements:**
- MOH-registered practitioners only
- Submit within 48 hours of return to work
- Supersedes general policy in Section 2.2

---

**Section 6: London Office Provisions**

**BANK HOLIDAY ALLOWANCE:**
- +8 additional days/year (UK bank holidays)
- Separate from standard 15-day annual leave
- **Total entitlement: 23 days** (15 annual + 8 bank holidays)
- Automatically added to leave balance

---

**Section 7 & 8: Attendance & Disciplinary Guidelines**

**Daily Requirements:**
- Check-in upon arrival
- Check-out before departure
- Accurate remote work logging

**MISSING CHECK-OUT POLICY:**
- Threshold: >5 instances in a calendar month
- Penalty: 2% salary deduction
- Counter resets monthly
- Repeated violations → formal warnings

**Purpose:** Accurate work hour tracking and labor regulation compliance

---

**Section 9: Special Circumstances**

**Maternity/Paternity Leave:**
- Governed by local regulations
- Contact regional HR for entitlements

**Extended Medical Leave:**
- Requires medical board certification
- Long-term disability insurance may apply

**Sabbatical Programs:**
- Available to 7+ years tenure employees
- Application process opens in January annually

---

## 3. RAG SYSTEM QUERY PATTERNS

### 3.1 Query Type Classification

**Type 1: Policy Lookup Queries**
- "What is the sick leave policy?"
- "How many annual leave days do I get?"
- "What are the Singapore MC requirements?"

**Resolution Path:** PDF → Extract relevant policy section

---

**Type 2: Employee-Specific Queries**
- "How many leave days does EMP1001 have?"
- "What is Sarah's tenure?"
- "Who is the manager for EMP1234?"

**Resolution Path:** CSV (employee master) → XLSX (leave balances) → PDF (policy rules)

---

**Type 3: Compliance Validation Queries**
- "Can EMP1001 take 5 days of sick leave?"
- "Does EMP1234 qualify for loyalty benefits?"
- "Has EMP1567 violated attendance policy?"

**Resolution Path:** 
1. CSV → Get employee details (location, joining_date)
2. XLSX → Check current balances
3. JSON → Check attendance violations (if applicable)
4. PDF → Validate against policy rules
5. Return compliance status

---

**Type 4: Cross-Referential Queries**
- "Which employees in Singapore have used sick leave without MCs?"
- "How many Engineering employees have loyalty tier 2 benefits?"
- "List employees with more than 5 missing check-outs this month"

**Resolution Path:** Multi-source join operations

---

### 3.2 Sample Query Resolution Workflows

#### Example 1: "How much annual leave does EMP1003 have?"

**Step 1:** Retrieve from `Available_Balances` sheet
```python
emp_id: EMP1003
annual_bal: 7
```

**Step 2:** Verify policy context from PDF
- Base entitlement: 15 days
- Check tenure for loyalty benefits

**Step 3:** Cross-reference with `employee_master.csv`
```python
joining_date: 2022-09-24
tenure: ~3.4 years (as of Feb 2026)
location: London
```

**Step 4:** Calculate entitlements
- Base: 15 days
- Loyalty Tier 1: +2 days (3+ years tenure)
- London Bank Holidays: +8 days
- **Total Entitlement: 25 days**
- **Used: 18 days** (25 - 7 remaining)

**Response:** "EMP1003 (Jerry Ramirez) has 7 days of annual leave remaining. As a London-based employee with 3.4 years of tenure, they are entitled to 25 total days annually (15 base + 2 loyalty tier 1 + 8 UK bank holidays)."

---

#### Example 2: "Is EMP1034 compliant with Singapore sick leave policy?"

**Step 1:** Retrieve employee details from CSV
```python
emp_id: EMP1034
name: Shirley Suarez
location: Singapore
```

**Step 2:** Check policy from PDF Section 5
- Singapore employees MUST provide MC for ALL sick leave
- This is a CRITICAL COMPLIANCE requirement

**Step 3:** Query leave history from XLSX
- Search `Leave_History` sheet for leave_type='Sick'
- Check status (Approved/Rejected)

**Step 4:** Validate MC submission
- **Data Limitation:** MC submission status not tracked in current data sources
- **Recommendation:** Need to query separate document management system

**Response:** "EMP1034 (Shirley Suarez) is a Singapore-based employee and is subject to mandatory MC requirements for ALL sick leave applications. To verify compliance, please check if MCs were submitted for all approved sick leave requests in the Leave_History sheet. This data is not available in the current dataset."

---

#### Example 3: "Which employees have attendance violations this month?"

**Step 1:** Define policy threshold from PDF Section 8
- Threshold: >5 missing check-outs in a calendar month

**Step 2:** Parse attendance JSON for current month (January 2026)
```python
for emp_id, data in attendance_logs.items():
    missing_checkouts = 0
    for record in data['records']:
        if record['date'].startswith('2026-01'):
            if record['check_out'] is None:
                missing_checkouts += 1
    
    if missing_checkouts > 5:
        violators.append(emp_id)
```

**Step 3:** Cross-reference with employee master for names and departments

**Step 4:** Calculate penalty
- 2% salary deduction per policy

**Response:** "[Generated list of violators with names, departments, and violation counts]"

---

## 4. DATA INTEGRITY & VALIDATION RULES

### 4.1 Cross-Source Validation

**Rule 1: Employee Existence**
- All `emp_id` values in JSON and XLSX must exist in CSV master file
- Orphaned records indicate data quality issues

**Rule 2: Active Status Filter**
- Only `is_active=True` employees should have current leave balances
- Inactive employees may have historical data only

**Rule 3: Leave Balance Consistency**
- Loyalty balance should only exist for employees with 3+ years tenure
- Sick balance should never exceed 10 days
- Emergency balance should never exceed 3 days

**Rule 4: Date Logical Consistency**
- `start_date` in leave history cannot be before `joining_date`
- Attendance dates cannot be before `joining_date`

---

### 4.2 Policy Application Logic

**Tenure Calculation:**
```python
from datetime import datetime

def calculate_tenure(joining_date_str, current_date_str='2026-02-05'):
    joining = datetime.strptime(joining_date_str, '%Y-%m-%d')
    current = datetime.strptime(current_date_str, '%Y-%m-%d')
    tenure_years = (current - joining).days / 365.25
    return tenure_years

def get_loyalty_tier(tenure_years):
    if tenure_years >= 5:
        return 'TIER_2', 5  # +5 days
    elif tenure_years >= 3:
        return 'TIER_1', 2  # +2 days
    else:
        return None, 0
```

**Location-Based Entitlements:**
```python
def calculate_total_annual_leave(location, tenure_years):
    base = 15
    loyalty = get_loyalty_tier(tenure_years)[1]
    
    if location == 'London':
        base += 8  # Bank holidays
    
    return base + loyalty
```

**Singapore MC Validation:**
```python
def requires_mc(location, leave_type):
    if location == 'Singapore' and leave_type == 'Sick':
        return True  # ALL sick leave requires MC
    elif leave_type == 'Sick' and duration > 2:
        return True  # Global policy: >2 days
    return False
```

---

## 5. SYSTEM LIMITATIONS & EDGE CASES

### 5.1 Known Data Gaps

**Gap 1: MC Submission Tracking**
- Current data sources do not track whether MCs were actually submitted
- Singapore compliance cannot be fully validated
- **Recommendation:** Integrate document management system

**Gap 2: Future Joining Dates**
- Some employees have joining_date in 2025-2026 (future dates)
- May indicate planned hires or data entry errors
- **Handling:** Treat as planned hires, tenure = 0

**Gap 3: Manager Approval Logic**
- `approver` field in leave history may not match `manager_id` in employee master
- Possible delegation or acting manager scenarios
- **Handling:** Accept both as valid

**Gap 4: Archived Data**
- ARCHIVED_2024 sheet exists but should not be used per instructions
- Historical trend analysis limited to current year

---

### 5.2 Edge Case Handling

**Edge Case 1: Breaks in Service**
- Policy states breaks in service reset tenure counter
- Current data does not track rehire dates
- **Assumption:** Treat `joining_date` as continuous service start

**Edge Case 2: Mid-Year Policy Changes**
- Policy effective date: January 1, 2026
- Leave requests before this date may follow old rules
- **Handling:** Apply current policy to all queries unless specified

**Edge Case 3: Partial Month Violations**
- "More than 5 missing check-outs in a calendar month"
- What if employee joined mid-month?
- **Handling:** Apply threshold as stated (absolute count, not prorated)

**Edge Case 4: Multiple Locations**
- Employee transferred between locations mid-year
- Current CSV shows only current location
- **Handling:** Apply policy based on current location field

---

## 6. BEST PRACTICES FOR RAG QUERIES

### 6.1 Query Resolution Checklist

**Before Answering Any Query:**
1. ✓ Identify which data sources are needed
2. ✓ Extract employee details from CSV if employee-specific
3. ✓ Cross-reference with policy PDF for rules
4. ✓ Validate against current balances in XLSX
5. ✓ Check for regional exceptions (Singapore, London)
6. ✓ Calculate tenure if loyalty benefits are involved
7. ✓ State any assumptions or limitations
8. ✓ Ground all statements in source data (no hallucinations)

---

### 6.2 Response Quality Standards

**Required Elements:**
- Source data citation (e.g., "According to Section 5 of the HR Policy Manual...")
- Specific values from data sources (not generic statements)
- Calculation methodology if applicable
- Any limitations or caveats

**Prohibited:**
- Invented information not in the data sources
- Assumptions stated as facts
- Generic policy descriptions without specific values
- Ignoring regional exceptions

---

### 6.3 Sample High-Quality Response Template

**Query:** "Can EMP1234 take 10 days of annual leave?"

**High-Quality Response:**
```
Based on the available data:

1. Employee Details (from employee_master.csv):
   - Name: [Name from CSV]
   - Location: [Location]
   - Joining Date: [Date]
   - Tenure: [Calculated] years

2. Leave Entitlements (from HR Policy Manual):
   - Base annual leave: 15 days
   - Loyalty bonus: [Tier 1/2/None] = [X days]
   - Regional bonus: [If London: +8 days]
   - Total entitlement: [Sum] days

3. Current Balance (from Available_Balances sheet):
   - Remaining: [annual_bal] days

4. Eligibility: [YES/NO]
   - Requested: 10 days
   - Available: [annual_bal] days
   - Conclusion: [Can/Cannot] take 10 days

5. Additional Requirements:
   - Must request 2 weeks in advance per policy
   - Manager approval required
   - Cannot carry forward beyond March 31 next year
```

---

## 7. DATA STATISTICS & INSIGHTS

### 7.1 Employee Distribution

**Total Employees:** 500+ (exact count available via CSV)

**Location Breakdown:**
- Singapore
- London
- Bangalore
- Tokyo
- Sydney
- New York
- Berlin
- [Other locations if any]

**Department Distribution:**
- Engineering
- Marketing
- Finance
- HR
- Legal
- Product
- IT
- Operations
- Customer Success

---

### 7.2 Leave Utilization Patterns

**From Leave_History Sheet:**
- Leave types: Annual, Sick, Loyalty, Emergency, Maternity, Paternity
- Status distribution: Approved, Rejected, Pending
- Approval rates vary by leave type

**Potential Analysis Queries:**
- Which department has highest sick leave usage?
- What is the approval rate for emergency leave?
- How many employees have utilized loyalty benefits?

---

### 7.3 Attendance Compliance

**From Attendance JSON:**
- Period covered: November 2025 - January 2026
- Daily records include check-in/check-out times
- Missing check-outs tracked per policy requirements

**Violation Detection:**
- Count null check_out values per employee per month
- Flag if count > 5 for salary deduction

---

## 8. INTEGRATION RECOMMENDATIONS

### 8.1 Additional Data Sources Needed

**High Priority:**
1. **Document Management System Integration**
   - Track MC submission for Singapore compliance
   - Store supporting documentation for leave requests

2. **Payroll System Link**
   - Automate salary deductions for attendance violations
   - Calculate loyalty bonus payments

3. **Calendar System**
   - Track regional holidays (Diwali, Golden Week, UK bank holidays)
   - Block out company-wide closure dates

---

### 8.2 Data Quality Improvements

**Recommendations:**
1. Add `mc_submitted` field to Leave_History sheet
2. Track location transfer history in employee master
3. Add `rehire_date` field for service break tracking
4. Implement real-time balance updates after leave approval
5. Add validation rules in Excel to prevent invalid data entry

---

### 8.3 RAG System Enhancements

**Phase 2 Features:**
1. **Predictive Analytics**
   - Forecast leave patterns by department/season
   - Predict attendance violation risks

2. **Automated Policy Checks**
   - Real-time eligibility validation at leave request time
   - Proactive compliance alerts for Singapore MC requirements

3. **Manager Dashboard Integration**
   - Show team leave calendar
   - Highlight policy violations
   - Streamline approval workflows

4. **Employee Self-Service**
   - Check leave balances via chatbot
   - Get personalized policy explanations
   - Submit pre-validated leave requests

---

## 9. SECURITY & COMPLIANCE

### 9.1 Data Classification

**Confidential & Proprietary:**
- HR Policy Manual (PDF)
- Employee Master (CSV) - Contains PII
- Leave Intelligence (XLSX) - Contains personal health information
- Attendance Logs (JSON) - Contains location tracking data

**Access Controls Required:**
- Role-based access (Employee, Manager, HR Admin)
- Employees can only view own data
- Managers can view direct reports
- HR admins have full access

---

### 9.2 Privacy Considerations

**PII Elements:**
- Employee names
- Email addresses
- Salary bands
- Medical leave records (implies health information)
- Location tracking data

**GDPR/Privacy Compliance:**
- Implement data minimization
- Provide data access/deletion rights
- Log all data access for audit
- Encrypt data at rest and in transit

---

## 10. CONCLUSION

This RAG-based HR Intelligence Bot system integrates four critical data sources to provide accurate, policy-compliant responses to HR queries. The system's strength lies in its ability to cross-reference multiple sources and apply complex business rules while maintaining data grounding.

### Key Success Factors:
1. **Accurate tenure calculations** for loyalty benefits
2. **Regional policy awareness** (Singapore MC, London bank holidays)
3. **Real-time balance validation** from Available_Balances sheet
4. **Attendance violation detection** from biometric logs
5. **Policy-first approach** using HR manual as authoritative source

### Critical Reminders:
- ⚠️ Never use ARCHIVED_2024 sheet
- ⚠️ Singapore employees require MC for ALL sick leave
- ⚠️ London employees get 23 days total annual leave (not 15)
- ⚠️ Loyalty benefits require exact tenure calculations
- ⚠️ Missing check-out threshold is >5/month (not ≥5)

### Ongoing Maintenance:
- Update policy PDF when rules change
- Refresh employee master monthly
- Archive old leave history annually
- Validate data quality quarterly
- Review regional policy changes semi-annually

---

**Document End**

For questions or updates to this documentation, contact:
- Global HR: global.hr@helixcorp.com
- System Administrator: [Contact via internal ticket]

---

*This documentation is based on data snapshot as of February 5, 2026*
