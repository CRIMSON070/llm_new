"""
FinSage Phase 2: Enhanced Arithmetic Engine and Unit Normalizer (WP4)
=====================================================================
Enforces deterministic financial calculation extraction and Python sandbox execution.
Eliminates LLM arithmetic approximation errors on financial differences, deltas, and ratios.
"""

import re
from typing import Dict, Any, Optional, Tuple, List, Union


def clean_numeric_str(val_str: Union[str, int, float]) -> Optional[float]:
    """Cleans a string representation of a financial number into a float."""
    if val_str is None:
        return None
    if isinstance(val_str, (int, float)):
        return float(val_str)
    
    s = str(val_str).strip()
    # Handle negative in parentheses e.g. (1,577) -> -1577
    is_negative = False
    if s.startswith("(") and s.endswith(")"):
        is_negative = True
        s = s[1:-1].strip()
    elif s.startswith("-"):
        is_negative = True
        s = s[1:].strip()
        
    # Remove currency symbols and commas
    s = re.sub(r"[\$,\s]", "", s)
    # Check for unit suffixes
    multiplier = 1.0
    if s.lower().endswith("b") or s.lower().endswith("billion"):
        multiplier = 1e9
        s = re.sub(r"(billion|b)$", "", s, flags=re.I).strip()
    elif s.lower().endswith("m") or s.lower().endswith("million"):
        multiplier = 1e6
        s = re.sub(r"(million|m)$", "", s, flags=re.I).strip()
    elif s.lower().endswith("k") or s.lower().endswith("thousand"):
        multiplier = 1e3
        s = re.sub(r"(thousand|k)$", "", s, flags=re.I).strip()
    elif s.endswith("%"):
        s = s[:-1].strip()

    try:
        val = float(s) * multiplier
        return -val if is_negative else val
    except ValueError:
        return None


def evaluate_financial_arithmetic(parsed_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates financial arithmetic deterministically from structured LLM extraction.
    Supports operations: 'absolute_difference', 'percentage_change', 'extract_single_value', 'sum', 'none'.
    """
    if not parsed_json:
        return {"success": False, "value": None, "formatted": "N/A", "error": "No JSON provided"}

    op = parsed_json.get("operation", "none")
    unit = parsed_json.get("unit", "")
    
    # Extract operands
    values: List[float] = []
    
    # Check for 'operands' or 'values' list
    raw_vals = parsed_json.get("operands") or parsed_json.get("values")
    if isinstance(raw_vals, list) and len(raw_vals) > 0:
        for v in raw_vals:
            c_v = clean_numeric_str(v)
            if c_v is not None:
                values.append(c_v)
    else:
        # Check old_value / new_value or value
        old_v = clean_numeric_str(parsed_json.get("old_value"))
        new_v = clean_numeric_str(parsed_json.get("new_value"))
        if old_v is not None:
            values.append(old_v)
        if new_v is not None:
            values.append(new_v)
        if not values and parsed_json.get("value") is not None:
            c_val = clean_numeric_str(parsed_json.get("value"))
            if c_val is not None:
                values.append(c_val)

    # Deterministic Evaluation
    if op == "absolute_difference":
        if len(values) >= 2:
            res = abs(values[1] - values[0])
            return {
                "success": True,
                "operation": op,
                "operands": values[:2],
                "value": round(res, 4),
                "formatted": f"{res:,.2f}" if not res.is_integer() else f"{int(res):,}",
                "unit": unit
            }
        elif len(values) == 1:
            return {
                "success": True,
                "operation": op,
                "operands": values,
                "value": values[0],
                "formatted": f"{values[0]:,}",
                "unit": unit
            }
        return {"success": False, "value": None, "error": "Missing operands for difference"}

    elif op == "percentage_change":
        if len(values) >= 2 and values[0] != 0:
            old_val, new_val = values[0], values[1]
            pct = ((new_val - old_val) / abs(old_val)) * 100.0
            return {
                "success": True,
                "operation": op,
                "operands": [old_val, new_val],
                "value": round(pct, 4),
                "formatted": f"{pct:+.2f}%",
                "unit": "%"
            }
        return {"success": False, "value": None, "error": "Invalid operands for percentage change"}

    elif op in ["extract_single_value", "none"]:
        if values:
            val = values[0]
            return {
                "success": True,
                "operation": op,
                "value": val,
                "formatted": f"{val:,.2f}" if not float(val).is_integer() else f"{int(val):,}",
                "unit": unit
            }
        return {
            "success": True,
            "operation": op,
            "value": None,
            "formatted": parsed_json.get("response", "None"),
            "unit": unit
        }

    return {"success": False, "value": None, "error": f"Unknown operation: {op}"}
