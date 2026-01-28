"""
schemas.py

This file contains ONLY schema definitions.
- No validation logic
- No functions
- No LLM-related code

It defines:
1. FULL_SCHEMA: the canonical database schema (single source of truth)
2. ROLE_TABLE_ACCESS: role-based permissions derived from the business needs
"""

# =========================
# 1. FULL CANONICAL SCHEMA
# =========================
# This must exactly reflect the real database structure.

FULL_SCHEMA = {
    "locations": [
        "id",
        "address",
        "city",
        "state",
        "zip_code",
        "region"
    ],

    "customers": [
        "id",
        "first_name",
        "last_name",
        "email",
        "phone",
        "location_id"
    ],

    "stores": [
        "id",
        "store_name",
        "store_type",      # online / physical
        "location_id"
    ],

    "categories": [
        "id",
        "category_name",
        "parent_category_id"
    ],

    "brands": [
        "id",
        "brand_name"
    ],

    "products": [
        "id",
        "name",
        "sku",
        "price",
        "stock_quantity",
        "category_id",
        "brand_id"
    ],

    "promotions": [
        "id",
        "promo_code",
        "discount_type",   # percentage / fixed
        "discount_value",
        "start_date",
        "end_date"
    ],

    "orders": [
        "id",
        "customer_id",
        "store_id",
        "order_date",
        "status",
        "total_amount"
    ],

    "order_items": [
        "id",
        "order_id",
        "product_id",
        "promotion_id",
        "quantity",
        "unit_price_at_sale"
    ]
}


# =========================
# 2. ROLE-BASED ACCESS MAP
# =========================
# This defines WHAT each role is allowed to see.
# It does NOT guarantee correctness — validation happens elsewhere.

ROLE_TABLE_ACCESS = {

    # ---- Sales role ----
    "sales": {
        "orders": [
            "id",
            "customer_id",
            "store_id",
            "order_date",
            "status",
            "total_amount"
        ],
        "order_items": [
            "order_id",
            "product_id",
            "quantity",
            "unit_price_at_sale"
        ],
        "customers": [
            "id",
            "first_name",
            "last_name",
            "email",
            "location_id"
        ],
        "products": [
            "id",
            "name",
            "price",
            "category_id",
            "brand_id"
        ],
        "stores": [
            "id",
            "store_name",
            "store_type"
        ]
    },

    # ---- Inventory / Supply Chain role ----
    "inventory": {
        "products": [
            "id",
            "name",
            "sku",
            "price",
            "stock_quantity",
            "category_id",
            "brand_id"
        ],
        "categories": [
            "id",
            "category_name",
            "parent_category_id"
        ],
        "brands": [
            "id",
            "brand_name"
        ],
        "stores": [
            "id",
            "store_name",
            "store_type"
        ]
    },

    # ---- Marketing role ----
    "marketing": {
        "promotions": [
            "id",
            "promo_code",
            "discount_type",
            "discount_value",
            "start_date",
            "end_date"
        ],
        "orders": [
            "id",
            "order_date",
            "total_amount"
        ],
        "order_items": [
            "order_id",
            "product_id",
            "quantity"
        ],
        "products": [
            "id",
            "name",
            "category_id",
            "brand_id"
        ]
    },

    # ---- Admin role (special case) ----
    # Admins can see everything, but execution is still controlled elsewhere.
    "admin": "ALL"
}
