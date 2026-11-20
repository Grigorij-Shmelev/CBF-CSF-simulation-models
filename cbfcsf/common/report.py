"""Report of pass/fail checks and informational values (the table the checks print)."""


class Report:
    def __init__(self):
        self.rows = []

    def check(self, name, value, lo, hi, unit=""):
        ok = bool(lo <= value <= hi)
        self.rows.append((name, value, lo, hi, unit, "PASS" if ok else "FAIL"))
        return ok

    def info(self, name, value, lo, hi, unit=""):
        inside = lo <= value <= hi
        self.rows.append((name, value, lo, hi, unit, "info (inside)" if inside else "info (OUTSIDE)"))

    def show(self):
        print(f"\n{'test':<46}{'value':>12}  {'bound':<24}{'':<8}result")
        for name, v, lo, hi, unit, res in self.rows:
            print(f"{name:<46}{v:>12.5g}  [{lo:.4g}, {hi:.4g}] {unit:<10}{res}")
        n_fail = sum(1 for r in self.rows if r[5] == "FAIL")
        n_info = sum(1 for r in self.rows if r[5].startswith("info"))
        print(f"\n{len(self.rows) - n_info} pass/fail checks, {n_fail} failed; {n_info} informational")
        return n_fail
