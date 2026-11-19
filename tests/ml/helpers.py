"""Synthetic source data for ML dataset unit tests."""

from __future__ import annotations

import pandas as pd


def synthetic_order_source() -> pd.DataFrame:
    """Return a small source snapshot covering eligibility edge cases."""

    rows = [
        {
            "order_id": "on_time",
            "order_status": "delivered",
            "purchase_timestamp": "2017-12-31 10:30:00",
            "delivered_timestamp": "2018-01-05 18:00:00",
            "estimated_delivery_timestamp": "2018-01-05 00:00:00",
            "warehouse_is_delayed": False,
            "customer_state": "sp",
            "customer_latitude": -23.55,
            "customer_longitude": -46.63,
            "customer_flag_has_gps": True,
            "flag_delivery_before_purchase": False,
        },
        {
            "order_id": "delayed",
            "order_status": "delivered",
            "purchase_timestamp": "2018-01-01 08:15:00",
            "delivered_timestamp": "2018-01-11 09:00:00",
            "estimated_delivery_timestamp": "2018-01-10 23:59:00",
            "warehouse_is_delayed": True,
            "customer_state": "rj",
            "customer_latitude": None,
            "customer_longitude": None,
            "customer_flag_has_gps": False,
            "flag_delivery_before_purchase": False,
        },
        {
            "order_id": "test_boundary",
            "order_status": "delivered",
            "purchase_timestamp": "2018-05-01 00:00:00",
            "delivered_timestamp": "2018-05-20 12:00:00",
            "estimated_delivery_timestamp": "2018-05-25 00:00:00",
            "warehouse_is_delayed": False,
            "customer_state": "mg",
            "customer_latitude": -19.92,
            "customer_longitude": -43.94,
            "customer_flag_has_gps": True,
            "flag_delivery_before_purchase": False,
        },
        {
            "order_id": "canceled",
            "order_status": "canceled",
            "purchase_timestamp": "2017-11-01 09:00:00",
            "delivered_timestamp": "2017-11-10 09:00:00",
            "estimated_delivery_timestamp": "2017-11-12 00:00:00",
            "warehouse_is_delayed": None,
            "customer_state": "SP",
            "customer_latitude": -23.55,
            "customer_longitude": -46.63,
            "customer_flag_has_gps": True,
            "flag_delivery_before_purchase": False,
        },
        {
            "order_id": "missing_delivery",
            "order_status": "delivered",
            "purchase_timestamp": "2017-10-01 09:00:00",
            "delivered_timestamp": None,
            "estimated_delivery_timestamp": "2017-10-15 00:00:00",
            "warehouse_is_delayed": None,
            "customer_state": "BA",
            "customer_latitude": -12.97,
            "customer_longitude": -38.50,
            "customer_flag_has_gps": True,
            "flag_delivery_before_purchase": False,
        },
        {
            "order_id": "missing_estimate",
            "order_status": "delivered",
            "purchase_timestamp": "2017-09-01 09:00:00",
            "delivered_timestamp": "2017-09-10 09:00:00",
            "estimated_delivery_timestamp": None,
            "warehouse_is_delayed": None,
            "customer_state": "PR",
            "customer_latitude": -25.43,
            "customer_longitude": -49.27,
            "customer_flag_has_gps": True,
            "flag_delivery_before_purchase": False,
        },
        {
            "order_id": "delivery_before_purchase",
            "order_status": "delivered",
            "purchase_timestamp": "2017-08-10 09:00:00",
            "delivered_timestamp": "2017-08-09 09:00:00",
            "estimated_delivery_timestamp": "2017-08-20 00:00:00",
            "warehouse_is_delayed": False,
            "customer_state": "SC",
            "customer_latitude": -27.59,
            "customer_longitude": -48.55,
            "customer_flag_has_gps": True,
            "flag_delivery_before_purchase": True,
        },
    ]
    return pd.DataFrame(rows)
