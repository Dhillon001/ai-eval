"""Toy parts-inventory MCP server for the ``configs/mcp_suite.yaml`` example.

A small, deterministic stdio MCP server (in-memory data, no network) that an agent can query to
answer questions about parts, stock levels and reservations. Run it directly with::

    python examples/mcp_inventory_server.py

or let ``ai-eval run --suite configs/mcp_suite.yaml`` launch it through the ``mcp`` adapter.

Tools:

- ``list_warehouses()``                          -> warehouse codes and cities
- ``search_parts(query, category=None)``         -> matching parts (sku, name, category)
- ``get_part(sku)``                              -> full part record (price, supplier, ...)
- ``check_stock(sku, warehouse=None)``           -> units on hand, per warehouse or in one
- ``reserve_part(sku, quantity, warehouse)``     -> reservation id; errors if stock is short

Unknown SKUs/warehouses and invalid quantities raise ``ToolError`` so the message reaches the
agent as a tool error and evals can check how it recovers. Reservations live only for the lifetime of the server process.

Requires the optional dependency: ``pip install "ai-eval[mcp]"``.
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

WAREHOUSES: dict[str, str] = {
    "SEA": "Seattle",
    "DAL": "Dallas",
    "BOS": "Boston",
}

PARTS: dict[str, dict] = {
    "BRK-001": {
        "sku": "BRK-001", "name": "ceramic brake pad set", "category": "brakes",
        "price_usd": 48.50, "supplier": "Stopwell", "lead_time_days": 3,
    },
    "BRK-014": {
        "sku": "BRK-014", "name": "vented brake rotor", "category": "brakes",
        "price_usd": 89.00, "supplier": "Stopwell", "lead_time_days": 5,
    },
    "FLT-204": {
        "sku": "FLT-204", "name": "oil filter", "category": "filters",
        "price_usd": 9.75, "supplier": "PureFlow", "lead_time_days": 2,
    },
    "FLT-310": {
        "sku": "FLT-310", "name": "cabin air filter", "category": "filters",
        "price_usd": 17.25, "supplier": "PureFlow", "lead_time_days": 2,
    },
    "BAT-075": {
        "sku": "BAT-075", "name": "12V AGM battery", "category": "electrical",
        "price_usd": 189.99, "supplier": "VoltCore", "lead_time_days": 7,
    },
    "SPK-110": {
        "sku": "SPK-110", "name": "iridium spark plug", "category": "ignition",
        "price_usd": 11.40, "supplier": "Ignis", "lead_time_days": 4,
    },
}

STOCK: dict[str, dict[str, int]] = {
    "BRK-001": {"SEA": 12, "DAL": 4, "BOS": 0},
    "BRK-014": {"SEA": 0, "DAL": 6, "BOS": 2},
    "FLT-204": {"SEA": 0, "DAL": 0, "BOS": 0},
    "FLT-310": {"SEA": 25, "DAL": 18, "BOS": 9},
    "BAT-075": {"SEA": 3, "DAL": 0, "BOS": 5},
    "SPK-110": {"SEA": 40, "DAL": 32, "BOS": 16},
}

_reservations: list[dict] = []

app = MCPServer("parts-inventory")


def _part(sku: str) -> dict:
    key = sku.strip().upper()
    if key not in PARTS:
        raise ToolError(f"unknown sku: {sku}")
    return PARTS[key]


def _warehouse(code: str) -> str:
    key = code.strip().upper()
    if key not in WAREHOUSES:
        raise ToolError(f"unknown warehouse: {code} (use one of {', '.join(WAREHOUSES)})")
    return key


@app.tool()
def list_warehouses() -> dict:
    """List warehouse codes and the city each one is in."""
    return {"warehouses": [{"code": c, "city": city} for c, city in WAREHOUSES.items()]}


@app.tool()
def search_parts(query: str, category: str | None = None) -> dict:
    """Search parts by name or SKU (case-insensitive substring), optionally within a category
    (brakes, filters, electrical, ignition)."""
    q = query.strip().lower()
    matches = [
        {"sku": p["sku"], "name": p["name"], "category": p["category"]}
        for p in PARTS.values()
        if (q in p["name"].lower() or q in p["sku"].lower())
        and (category is None or p["category"] == category.strip().lower())
    ]
    return {"results": matches}


@app.tool()
def get_part(sku: str) -> dict:
    """Get the full record for a part: name, category, unit price (USD), supplier, lead time."""
    return dict(_part(sku))


@app.tool()
def check_stock(sku: str, warehouse: str | None = None) -> dict:
    """Units on hand for a part, in every warehouse or only in the given warehouse code."""
    part = _part(sku)
    levels = STOCK[part["sku"]]
    if warehouse is not None:
        code = _warehouse(warehouse)
        return {"sku": part["sku"], "warehouse": code, "on_hand": levels[code]}
    return {"sku": part["sku"], "on_hand": dict(levels), "total": sum(levels.values())}


@app.tool()
def reserve_part(sku: str, quantity: int, warehouse: str) -> dict:
    """Reserve units of a part in one warehouse. Fails if the warehouse has too few on hand."""
    part = _part(sku)
    code = _warehouse(warehouse)
    if quantity < 1:
        raise ToolError("quantity must be at least 1")
    on_hand = STOCK[part["sku"]][code]
    if quantity > on_hand:
        raise ToolError(f"insufficient stock: {on_hand} {part['sku']} on hand in {code}, {quantity} requested")
    STOCK[part["sku"]][code] = on_hand - quantity
    reservation = {
        "reservation_id": f"R-{len(_reservations) + 1:04d}",
        "sku": part["sku"],
        "warehouse": code,
        "quantity": quantity,
    }
    _reservations.append(reservation)
    return reservation


if __name__ == "__main__":
    app.run()
