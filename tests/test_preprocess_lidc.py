"""Unit tests for LIDC XML parse + physical-nodule clustering (no DICOM required)."""

from __future__ import annotations

import numpy as np

from pathlib import Path

from pulmoscan.data.preprocess_lidc import (
    ReaderNodule,
    SeriesInfo,
    cluster_physical_nodules,
    parse_lidc_xml,
    select_series_for_training,
)


def test_cluster_physical_nodules_merges_nearby_readers():
    nodules = [
        ReaderNodule(0, "a", 5.0, [10.0, 20.0, 30.0], characteristics={"malignancy": 5}),
        ReaderNodule(1, "b", 4.0, [11.0, 20.5, 30.2], characteristics={"malignancy": 4}),
        ReaderNodule(2, "c", 1.0, [80.0, 80.0, 80.0], characteristics={"malignancy": 1}),
    ]
    phys = cluster_physical_nodules(nodules, distance_mm=6.0)
    assert len(phys) == 2
    sizes = sorted(p["num_observations"] for p in phys)
    assert sizes == [1, 2]
    merged = max(phys, key=lambda p: p["num_observations"])
    assert merged["confidence"] == "medium"
    assert abs(merged["malignancy_mean"] - 4.5) < 1e-6


def test_parse_lidc_xml_fixture(tmp_path):
    import SimpleITK as sitk

    # Identity geometry: index (x,y,z) == patient mm
    img = sitk.Image(64, 64, 64, sitk.sitkFloat32)
    img.SetSpacing((1.0, 1.0, 1.0))
    img.SetOrigin((0.0, 0.0, 0.0))

    xml = """<?xml version="1.0"?>
    <LidcReadMessage>
      <ResponseHeader>
        <SeriesInstanceUid>1.2.3.test</SeriesInstanceUid>
      </ResponseHeader>
      <readingSession>
        <unblindedReadNodule>
          <noduleID>Nodule 001</noduleID>
          <characteristics><malignancy>5</malignancy></characteristics>
          <roi>
            <imageZposition>30.0</imageZposition>
            <edgeMap><xCoord>10</xCoord><yCoord>20</yCoord></edgeMap>
            <edgeMap><xCoord>12</xCoord><yCoord>22</yCoord></edgeMap>
          </roi>
        </unblindedReadNodule>
      </readingSession>
      <readingSession>
        <unblindedReadNodule>
          <noduleID>Nodule 002</noduleID>
          <characteristics><malignancy>4</malignancy></characteristics>
          <roi>
            <imageZposition>30.0</imageZposition>
            <edgeMap><xCoord>11</xCoord><yCoord>21</yCoord></edgeMap>
          </roi>
        </unblindedReadNodule>
      </readingSession>
    </LidcReadMessage>
    """
    path = tmp_path / "case.xml"
    path.write_text(xml, encoding="utf-8")
    doc, nodules = parse_lidc_xml(path, img)
    assert doc["series_instance_uid"] == "1.2.3.test"
    assert len(nodules) == 2
    assert nodules[0].malignancy == 5.0
    assert len(nodules[0].centroid_patient_mm) == 3
    # mean of (10,20) and (12,22) at z=30 → ~(11, 21, 30)
    assert abs(nodules[0].centroid_patient_mm[0] - 11.0) < 1e-3
    assert abs(nodules[0].centroid_patient_mm[1] - 21.0) < 1e-3
    assert abs(nodules[0].centroid_patient_mm[2] - 30.0) < 1e-3


def test_select_series_prefers_annotated_chest_over_angio(tmp_path):
    xml = tmp_path / "a.xml"
    xml.write_text("<x/>", encoding="utf-8")
    chest = SeriesInfo(
        "uid-chest",
        "LIDC-IDRI-0001",
        tmp_path / "chest",
        n_files=120,
        series_description="CHEST",
        path_hint=str(tmp_path / "LIDC-IDRI-0001" / "chest"),
    )
    angio = SeriesInfo(
        "uid-angio",
        "LIDC-IDRI-0001",
        tmp_path / "angio",
        n_files=200,
        series_description="CTA ANGIO",
        path_hint=str(tmp_path / "LIDC-IDRI-0001" / "angio"),
    )
    other = SeriesInfo(
        "uid-other",
        "LIDC-IDRI-0002",
        tmp_path / "other",
        n_files=80,
        path_hint=str(tmp_path / "LIDC-IDRI-0002" / "chest"),
    )
    selected, stats = select_series_for_training(
        [chest, angio, other],
        xml_index={"uid-chest": xml},
        annotated_only=True,
        skip_angio=True,
        one_per_patient=True,
    )
    assert [s.series_uid for s in selected] == ["uid-chest"]
    assert stats["dropped_no_xml"] >= 1
