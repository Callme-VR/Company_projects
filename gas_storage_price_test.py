from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence


DateLike = str | date | datetime
Number = int | float


@dataclass(frozen=True)
class CashFlowEvent:
    event_date: date
    kind: str
    requested_volume: float
    executed_volume: float
    price: float
    cash_flow: float
    inventory_after: float


@dataclass(frozen=True)
class ContractValue:
    value: float
    purchase_cost: float
    sale_proceeds: float
    storage_cost: float
    fixed_storage_cost: float
    ending_inventory: float
    events: tuple[CashFlowEvent, ...]


def _to_date(d: DateLike) -> date:
    if isinstance(d, str):
        return datetime.strptime(d, "%Y-%m-%d").date()
    if isinstance(d, datetime):
        return d.date()
    return d


class PriceCurve:
    """Daily linear interpolation over a dated commodity price series."""

    def __init__(self, observations: Iterable[tuple[DateLike, Number]]) -> None:
        points = sorted((_to_date(d), float(p)) for d, p in observations)
        if not points:
            raise ValueError("PriceCurve requires at least one observation.")
        self._dates = tuple(d for d, _ in points)
        self._prices = tuple(p for _, p in points)

    @classmethod
    def from_csv(
        cls,
        csv_path: str | Path,
        date_column: str = "Dates",
        price_column: str = "Prices",
    ) -> "PriceCurve":
        with Path(csv_path).open(newline="") as handle:
            reader = csv.DictReader(handle)
            return cls((row[date_column], float(row[price_column])) for row in reader)

    def price_on(self, pricing_date: DateLike) -> float:
        target = _to_date(pricing_date)
        if target < self._dates[0] or target > self._dates[-1]:
            raise ValueError(
                f"{target.isoformat()} is outside the available price curve "
                f"({self._dates[0].isoformat()} to {self._dates[-1].isoformat()}). "
                "Pass an explicit price for this date or extend the curve."
            )

        if target in self._dates:
            return self._prices[self._dates.index(target)]

        right = next(i for i, curve_date in enumerate(self._dates) if curve_date > target)
        left = right - 1
        left_date = self._dates[left]
        right_date = self._dates[right]
        left_price = self._prices[left]
        right_price = self._prices[right]
        total_days = (right_date - left_date).days
        elapsed_days = (target - left_date).days
        return left_price + (right_price - left_price) * elapsed_days / total_days


def price_storage_contract(
    injection_dates: Sequence[DateLike],
    withdrawal_dates: Sequence[DateLike],
    purchase_prices: Mapping[DateLike, Number] | Sequence[Number] | Number | None = None,
    sale_prices: Mapping[DateLike, Number] | Sequence[Number] | Number | None = None,
    injection_rate: Number = float("inf"),
    withdrawal_rate: Number = float("inf"),
    max_storage_volume: Number = float("inf"),
    storage_cost_per_unit_per_day: Number = 0.0,
    injection_volumes: Mapping[DateLike, Number] | Sequence[Number] | Number | None = None,
    withdrawal_volumes: Mapping[DateLike, Number] | Sequence[Number] | Number | None = None,
    fixed_storage_cost: Number = 0.0,
    price_curve: PriceCurve | None = None,
    allow_partial: bool = False,
) -> ContractValue:
    """
    Value a physical gas storage contract with dated injections and withdrawals.

    Cash flows are valued with zero interest rates:
    value = withdrawal sale proceeds - injection purchase costs - storage costs.
    """
    inj_dates = [_to_date(d) for d in injection_dates]
    wth_dates = [_to_date(d) for d in withdrawal_dates]
    all_dates = sorted(set(inj_dates + wth_dates))

    def _resolve(
        raw: Mapping[DateLike, Number] | Sequence[Number] | Number | None,
        dates: list[date],
        default: Number = 0.0,
    ) -> dict[date, float]:
        if raw is None:
            return {d: float(default) for d in dates}
        if isinstance(raw, (int, float)):
            return {d: float(raw) for d in dates}
        if isinstance(raw, Mapping):
            return {_to_date(k): float(v) for k, v in raw.items()}
        seq = list(raw)
        return {d: float(seq[i]) for i, d in enumerate(dates)}

    inj_vol = _resolve(injection_volumes, inj_dates, 1.0)
    wth_vol = _resolve(withdrawal_volumes, wth_dates, 1.0)

    def _resolve_prices(
        raw: Mapping[DateLike, Number] | Sequence[Number] | Number | None,
        dates: list[date],
    ) -> dict[date, float]:
        if raw is None:
            return {}
        if isinstance(raw, (int, float)):
            return {d: float(raw) for d in dates}
        if isinstance(raw, Mapping):
            return {_to_date(k): float(v) for k, v in raw.items()}
        seq = list(raw)
        return {d: float(seq[i]) for i, d in enumerate(dates)}

    purch = _resolve_prices(purchase_prices, inj_dates)
    sales = _resolve_prices(sale_prices, wth_dates)

    inventory = 0.0
    events: list[CashFlowEvent] = []
    prev_date = all_dates[0]
    variable_storage_cost = 0.0

    for d in all_dates:
        days_held = (d - prev_date).days if prev_date else 0
        if days_held > 0 and inventory > 0:
            sc = inventory * storage_cost_per_unit_per_day * days_held
            variable_storage_cost += sc
            events.append(CashFlowEvent(d, "storage", 0.0, 0.0, 0.0, -sc, inventory))

        if d in inj_vol:
            vol = inj_vol[d]
            rate = float(injection_rate)
            if rate != float("inf") and vol > rate:
                if allow_partial:
                    vol = rate
                else:
                    raise ValueError(f"Injection volume {vol} exceeds rate {rate} on {d}")
            if inventory + vol > max_storage_volume:
                if allow_partial:
                    vol = max_storage_volume - inventory
                else:
                    raise ValueError(
                        f"Injection on {d} would exceed max storage {max_storage_volume}"
                    )
            price = purch.get(d, price_curve.price_on(d) if price_curve else 0.0)
            cash = -vol * price
            inventory += vol
            events.append(CashFlowEvent(d, "injection", inj_vol[d], vol, price, cash, inventory))

        if d in wth_vol:
            vol = wth_vol[d]
            rate = float(withdrawal_rate)
            if rate != float("inf") and vol > rate:
                if allow_partial:
                    vol = rate
                else:
                    raise ValueError(f"Withdrawal volume {vol} exceeds rate {rate} on {d}")
            if vol > inventory:
                if allow_partial:
                    vol = inventory
                else:
                    raise ValueError(
                        f"Withdrawal on {d} exceeds inventory {inventory}"
                    )
            price = sales.get(d, price_curve.price_on(d) if price_curve else 0.0)
            cash = vol * price
            inventory -= vol
            events.append(CashFlowEvent(d, "withdrawal", wth_vol[d], vol, price, cash, inventory))

        prev_date = d

    total_storage = variable_storage_cost + float(fixed_storage_cost)
    total_purchase = sum(-e.cash_flow for e in events if e.kind == "injection")
    total_sale = sum(e.cash_flow for e in events if e.kind == "withdrawal")
    net_value = total_sale - total_purchase - total_storage

    return ContractValue(
        value=round(net_value, 2),
        purchase_cost=round(total_purchase, 2),
        sale_proceeds=round(total_sale, 2),
        storage_cost=round(total_storage, 2),
        fixed_storage_cost=round(float(fixed_storage_cost), 2),
        ending_inventory=round(inventory, 2),
        events=tuple(events),
    )
