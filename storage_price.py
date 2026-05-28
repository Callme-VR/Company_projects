from gas_storage_price_test import PriceCurve, price_storage_contract, ContractValue


def main() -> None:
    curve = PriceCurve.from_csv("Nat_Gas.csv")

    value: ContractValue = price_storage_contract(
        injection_dates=["2024-01-01", "2024-02-01", "2024-03-01"],
        withdrawal_dates=["2024-06-01", "2024-07-01", "2024-08-01"],
        injection_volumes=1000,
        withdrawal_volumes=1000,
        injection_rate=500,
        withdrawal_rate=500,
        max_storage_volume=3000,
        storage_cost_per_unit_per_day=0.001,
        fixed_storage_cost=10000,
        price_curve=curve,
    )

    print(f"Contract value: ${value.value:,.2f}")
    print(f"  Purchase cost:  ${value.purchase_cost:,.2f}")
    print(f"  Sale proceeds:  ${value.sale_proceeds:,.2f}")
    print(f"  Storage cost:   ${value.storage_cost:,.2f}")
    print(f"  End inventory:  {value.ending_inventory:,.0f}")
    print(f"\nEvents ({len(value.events)}):")
    for ev in value.events:
        print(
            f"  {ev.event_date}  {ev.kind:>10s}  "
            f"vol={ev.executed_volume:>6.0f}  "
            f"price=${ev.price:>6.2f}  "
            f"cash=${ev.cash_flow:>8.2f}  "
            f"inv={ev.inventory_after:>6.0f}"
        )


if __name__ == "__main__":
    main()
