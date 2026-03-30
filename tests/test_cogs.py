"""
Unit tests for the COGS cost engine — no network, no DB, no AI.
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.cogs import MATERIALS, PROCESSES, REGIONS, VOLUME_TIERS, compute_cogs


def _make_geom(volume_cm3=10.0, complexity=3.0, is_assembly=False, confidence_interval=0.15):
    return SimpleNamespace(
        volume_cm3=volume_cm3,
        complexity_score=complexity,
        is_assembly=is_assembly,
        confidence_interval=confidence_interval,
        surface_area_cm2=volume_cm3 * 2,
        bounding_box_mm=SimpleNamespace(x=50, y=50, z=50),
    )


class TestSubtractiveProcesses:
    def test_cnc3axis_returns_all_regions(self):
        geom = _make_geom()
        result = compute_cogs(geom, "Al6061-T6", "CNC_3axis")
        assert set(result["regions"].keys()) == set(REGIONS.keys())

    def test_cnc5axis_has_higher_total_than_3axis(self):
        geom = _make_geom()
        r3 = compute_cogs(geom, "Al6061-T6", "CNC_3axis")
        r5 = compute_cogs(geom, "Al6061-T6", "CNC_5axis")
        # 5-axis has higher setup + lower MRR → higher cost at proto qty
        assert r5["regions"]["local"]["tiers"]["1"]["total"] > r3["regions"]["local"]["tiers"]["1"]["total"]

    def test_volume_discount_applied(self):
        geom = _make_geom()
        result = compute_cogs(geom, "Al6061-T6", "CNC_3axis")
        local = result["regions"]["local"]["tiers"]
        # qty=1000 material should be cheaper than qty=1 material
        assert local["1000"]["material"] < local["1"]["material"]

    def test_confidence_interval_bounds(self):
        geom = _make_geom(confidence_interval=0.20)
        result = compute_cogs(geom, "Al6061-T6", "CNC_3axis")
        for tier_data in result["regions"]["local"]["tiers"].values():
            assert tier_data["low"] < tier_data["total"] < tier_data["high"]

    def test_all_volume_tiers_present(self):
        geom = _make_geom()
        result = compute_cogs(geom, "Al6061-T6", "CNC_3axis")
        for qty in VOLUME_TIERS:
            assert str(qty) in result["regions"]["local"]["tiers"]

    def test_turning_process(self):
        geom = _make_geom()
        result = compute_cogs(geom, "Al6061-T6", "Turning")
        assert result["process_type"] == "subtractive"
        assert result["regions"]["local"]["tiers"]["1"]["total"] > 0

    def test_titanium_more_expensive_than_aluminium(self):
        geom = _make_geom()
        al = compute_cogs(geom, "Al6061-T6", "CNC_3axis")
        ti = compute_cogs(geom, "Titanium_Grade5", "CNC_3axis")
        assert ti["regions"]["local"]["tiers"]["1"]["total"] > al["regions"]["local"]["tiers"]["1"]["total"]


class TestAdditiveProcesses:
    def test_fdm_process_type(self):
        geom = _make_geom()
        result = compute_cogs(geom, "PLA", "FDM")
        assert result["process_type"] == "additive"

    def test_sla_process_type(self):
        geom = _make_geom()
        result = compute_cogs(geom, "Standard_Resin", "SLA")
        assert result["process_type"] == "additive"

    def test_sls_process_type(self):
        geom = _make_geom()
        result = compute_cogs(geom, "Nylon_PA12", "SLS")
        assert result["process_type"] == "additive"

    def test_fdm_cheaper_at_proto_than_cnc(self):
        geom = _make_geom()
        fdm = compute_cogs(geom, "PLA", "FDM")
        cnc = compute_cogs(geom, "Al6061-T6", "CNC_3axis")
        # FDM at 1 unit should be cheaper than CNC at 1 unit for simple parts
        assert fdm["regions"]["local"]["tiers"]["1"]["total"] < cnc["regions"]["local"]["tiers"]["1"]["total"]

    def test_additive_no_stock_volume_inflation(self):
        geom = _make_geom(volume_cm3=10.0)
        result = compute_cogs(geom, "PLA", "FDM")
        # Stock volume for additive ≈ part volume × support factor (≤ 1.15×)
        assert result["stock_volume_cm3"] <= geom.volume_cm3 * 1.20


class TestMoldingProcesses:
    def test_injection_molding_process_type(self):
        geom = _make_geom()
        result = compute_cogs(geom, "PP", "InjectionMolding")
        assert result["process_type"] == "molding"

    def test_tooling_cost_present(self):
        geom = _make_geom()
        result = compute_cogs(geom, "PP", "InjectionMolding")
        assert "tooling_cost_usd" in result
        assert result["tooling_cost_usd"] > 0

    def test_molding_expensive_at_proto_cheap_at_volume(self):
        geom = _make_geom()
        result = compute_cogs(geom, "PP", "InjectionMolding")
        local = result["regions"]["local"]["tiers"]
        # Tooling amortisation makes qty=1 very expensive
        assert local["1"]["total"] > local["1000"]["total"] * 5

    def test_molding_has_tooling_field_in_tiers(self):
        geom = _make_geom()
        result = compute_cogs(geom, "PP", "InjectionMolding")
        tier = result["regions"]["local"]["tiers"]["1"]
        assert "tooling" in tier


class TestMaterialFallback:
    def test_unknown_material_falls_back_to_al6061(self):
        geom = _make_geom()
        result = compute_cogs(geom, "NonExistentMaterial", "CNC_3axis")
        assert result["material_grade"] == MATERIALS["Al6061-T6"]["name"]

    def test_unknown_process_falls_back_to_cnc3axis(self):
        geom = _make_geom()
        result = compute_cogs(geom, "Al6061-T6", "NonExistentProcess")
        assert result["process"] == PROCESSES["CNC_3axis"]["name"]
