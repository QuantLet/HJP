from __future__ import annotations

import argparse
import csv
import pickle
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.core.multiarray import _reconstruct as _numpy_reconstruct
from numpy.core.multiarray import scalar as _numpy_scalar


DEFAULT_PICKLE = Path(__file__).with_name("Q_results_refined.pickle")


class _StoredHawkesJDParams:
    """Passive target for the attribute dictionary stored in the pickle."""


def _stored_tensor(value: Any) -> Any:
    """Replace TensorFlow's ``convert_to_tensor`` during deserialization."""

    return value


class _ResultsUnpickler(pickle.Unpickler):
    """Load the known result format without importing TensorFlow or the model."""

    _ALLOWED_GLOBALS = {
        ("numpy.core.multiarray", "_reconstruct"): _numpy_reconstruct,
        ("numpy._core.multiarray", "_reconstruct"): _numpy_reconstruct,
        ("numpy.core.multiarray", "scalar"): _numpy_scalar,
        ("numpy._core.multiarray", "scalar"): _numpy_scalar,
        ("numpy", "ndarray"): np.ndarray,
        ("numpy", "dtype"): np.dtype,
        (
            "stochvolmodels.pricers.hawkes_jd_pricer",
            "HawkesJDParams",
        ): _StoredHawkesJDParams,
        (
            "tensorflow.python.framework.ops",
            "convert_to_tensor",
        ): _stored_tensor,
    }

    def find_class(self, module: str, name: str) -> Any:
        try:
            return self._ALLOWED_GLOBALS[(module, name)]
        except KeyError as exc:
            raise pickle.UnpicklingError(
                f"unsupported global in results pickle: {module}.{name}"
            ) from exc


@dataclass(frozen=True)
class DailyDiagnostics:
    calibration_date: str
    m_plus: float
    m_minus: float
    spectral_abscissa: float


@dataclass(frozen=True)
class SummaryRow:
    condition: str
    diagnostic: str
    requirement: str
    minimum: float
    median: float
    maximum: float
    holds: int
    total: int
    signed_values: bool = False


def load_results(path: Path) -> Mapping[str, Mapping[str, Any]]:
    """Load and minimally validate the dated calibration-result mapping."""

    with path.open("rb") as file:
        results = _ResultsUnpickler(file).load()

    if not isinstance(results, Mapping) or not results:
        raise ValueError("expected a non-empty mapping keyed by calibration date")
    return results


def _real_scalar(value: Any, *, field: str, calibration_date: str) -> float:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(
            f"{calibration_date}: {field} must be scalar, got shape {array.shape}"
        )

    scalar = complex(array.reshape(()).item())
    if abs(scalar.imag) > 1e-10:
        raise ValueError(
            f"{calibration_date}: {field} has a non-negligible imaginary part"
        )
    if not np.isfinite(scalar.real):
        raise ValueError(f"{calibration_date}: {field} is not finite")
    return float(scalar.real)


def compute_daily_diagnostics(
    results: Mapping[str, Mapping[str, Any]],
) -> list[DailyDiagnostics]:
    """Compute C1 margins and the sharp C2 spectral condition for every row."""

    diagnostics: list[DailyDiagnostics] = []
    required_fields = (
        "nu_p",
        "eta_p",
        "nu_m",
        "eta_m",
        "kappa_p",
        "kappa_m",
        "beta11",
        "beta12",
        "beta21",
        "beta22",
    )

    for calibration_date in sorted(results):
        try:
            Date.fromisoformat(calibration_date)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid ISO calibration date: {calibration_date!r}") from exc

        row = results[calibration_date]
        if not isinstance(row, Mapping):
            raise ValueError(f"{calibration_date}: expected a result mapping")

        try:
            measure_change = np.asarray(row["measure_change_params"], dtype=float)
            params = row["measure_change_results"]
        except KeyError as exc:
            raise ValueError(
                f"{calibration_date}: missing required result key {exc.args[0]!r}"
            ) from exc

        if measure_change.shape != (3,) or not np.all(np.isfinite(measure_change)):
            raise ValueError(
                f"{calibration_date}: measure_change_params must be three finite "
                "values [sigma, chi_p, chi_m]"
            )
        _, chi_plus, chi_minus = measure_change

        values: dict[str, float] = {}
        for field in required_fields:
            if not hasattr(params, field):
                raise ValueError(
                    f"{calibration_date}: measure_change_results lacks {field!r}"
                )
            values[field] = _real_scalar(
                getattr(params, field),
                field=field,
                calibration_date=calibration_date,
            )

        # The pickle stores eta^{+,Q} and eta^{-,Q}; invert the measure-change
        # formulas to obtain the original C1 margins without P_params.csv.
        plus_denominator = 1.0 + values["eta_p"] * chi_plus
        minus_denominator = 1.0 - values["eta_m"] * chi_minus
        if plus_denominator == 0.0 or minus_denominator == 0.0:
            raise ValueError(f"{calibration_date}: singular jump-size transformation")
        m_plus = 1.0 / plus_denominator
        m_minus = 1.0 / minus_denominator

        mean_jump_plus_q = values["nu_p"] + values["eta_p"]
        mean_jump_minus_q = values["nu_m"] - values["eta_m"]
        phi_q = np.array(
            [
                [
                    -values["kappa_p"]
                    + values["beta11"] * mean_jump_plus_q,
                    values["beta12"] * mean_jump_minus_q,
                ],
                [
                    values["beta21"] * mean_jump_plus_q,
                    -values["kappa_m"]
                    + values["beta22"] * mean_jump_minus_q,
                ],
            ],
            dtype=float,
        )
        spectral_abscissa = float(np.max(np.real(np.linalg.eigvals(phi_q))))

        diagnostics.append(
            DailyDiagnostics(
                calibration_date=calibration_date,
                m_plus=m_plus,
                m_minus=m_minus,
                spectral_abscissa=spectral_abscissa,
            )
        )

    return diagnostics


def summarize(diagnostics: Sequence[DailyDiagnostics]) -> list[SummaryRow]:
    if not diagnostics:
        raise ValueError("cannot summarize an empty diagnostic sequence")

    def make_row(
        condition: str,
        diagnostic: str,
        requirement: str,
        getter: Callable[[DailyDiagnostics], float],
        test: Callable[[float], bool],
        *,
        signed_values: bool = False,
    ) -> SummaryRow:
        values = np.asarray([getter(item) for item in diagnostics], dtype=float)
        return SummaryRow(
            condition=condition,
            diagnostic=diagnostic,
            requirement=requirement,
            minimum=float(np.min(values)),
            median=float(np.median(values)),
            maximum=float(np.max(values)),
            holds=sum(test(float(value)) for value in values),
            total=len(values),
            signed_values=signed_values,
        )

    return [
        make_row(
            "(C1)",
            "m+ = 1 - eta+ chi+",
            "> 0",
            lambda item: item.m_plus,
            lambda value: value > 0.0,
        ),
        make_row(
            "(C1)",
            "m- = 1 + eta- chi-",
            "> 0",
            lambda item: item.m_minus,
            lambda value: value > 0.0,
        ),
        make_row(
            "(C2)",
            "s(PhiQ) (spectral abscissa)",
            "< 0",
            lambda item: item.spectral_abscissa,
            lambda value: value < 0.0,
            signed_values=True,
        ),
    ]


def _format_value(value: float, *, signed: bool) -> str:
    return f"{value:+.2f}" if signed else f"{value:.2f}"


def render_markdown(summary: Sequence[SummaryRow]) -> str:
    lines = [
        "| Condition | Diagnostic | Req. | Min | Median | Max | Holds |",
        "|:--|:--|:--:|--:|--:|--:|--:|",
    ]
    for row in summary:
        lines.append(
            "| {condition} | {diagnostic} | {requirement} | {minimum} | "
            "{median} | {maximum} | {holds}/{total} |".format(
                condition=row.condition,
                diagnostic=row.diagnostic,
                requirement=row.requirement,
                minimum=_format_value(row.minimum, signed=row.signed_values),
                median=_format_value(row.median, signed=row.signed_values),
                maximum=_format_value(row.maximum, signed=row.signed_values),
                holds=row.holds,
                total=row.total,
            )
        )
    return "\n".join(lines)


def write_summary_csv(path: Path, summary: Sequence[SummaryRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Condition", "Diagnostic", "Req.", "Min", "Median", "Max", "Holds"]
        )
        for row in summary:
            writer.writerow(
                [
                    row.condition,
                    row.diagnostic,
                    row.requirement,
                    f"{row.minimum:.15g}",
                    f"{row.median:.15g}",
                    f"{row.maximum:.15g}",
                    f"{row.holds}/{row.total}",
                ]
            )


def write_daily_csv(path: Path, diagnostics: Sequence[DailyDiagnostics]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["date", "m_plus", "m_minus", "spectral_abscissa", "c2_holds"])
        for item in diagnostics:
            writer.writerow(
                [
                    item.calibration_date,
                    f"{item.m_plus:.15g}",
                    f"{item.m_minus:.15g}",
                    f"{item.spectral_abscissa:.15g}",
                    item.spectral_abscissa < 0.0,
                ]
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pickle_path",
        nargs="?",
        type=Path,
        default=DEFAULT_PICKLE,
        help=f"results pickle (default: {DEFAULT_PICKLE})",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        help="optionally write the unrounded three-row summary as CSV",
    )
    parser.add_argument(
        "--daily-csv",
        type=Path,
        help="optionally write all per-date diagnostics as CSV",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results = load_results(args.pickle_path)
    diagnostics = compute_daily_diagnostics(results)
    summary = summarize(diagnostics)

    print(render_markdown(summary))
    c2 = summary[-1]
    c2_percentage = 100.0 * c2.holds / c2.total
    failures_by_year = Counter(
        item.calibration_date[:4]
        for item in diagnostics
        if item.spectral_abscissa >= 0.0
    )
    print()
    print(
        "Verification across "
        f"{len(diagnostics)} calibration dates "
        f"({diagnostics[0].calibration_date} to {diagnostics[-1].calibration_date})."
    )
    print(
        f"C1 holds on every date. C2 holds on {c2.holds}/{c2.total} "
        f"dates ({c2_percentage:.1f}%); failures by year: "
        + ", ".join(f"{year}={count}" for year, count in sorted(failures_by_year.items()))
        + "."
    )

    if args.summary_csv:
        write_summary_csv(args.summary_csv, summary)
    if args.daily_csv:
        write_daily_csv(args.daily_csv, diagnostics)


if __name__ == "__main__":
    main()
