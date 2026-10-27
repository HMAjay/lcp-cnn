"""Unit tests for LIDC XML parse + physical-nodule clustering (no DICOM required)."""

from __future__ import annotations

import numpy as np

from pulmoscan.data.preprocess_lidc import (
    ReaderNodule,
    cluster_physical_nodules,
    parse_lidc_xml,
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
            <imageZposition>30.5</imageZposition>
            <edgeMap><xCoord>11</xCoord><yCoord>21</yCoord></edgeMap>
          </roi>
        </unblindedReadNodule>
      </readingSession>
    </LidcReadMessage>
    """
    path = tmp_path / "case.xml"
    path.write_text(xml, encoding="utf-8")
    affine = np.eye(4, dtype=np.float64)
    doc, nodules = parse_lidc_xml(path, affine)
    assert doc["series_instance_uid"] == "1.2.3.test"
    assert len(nodules) == 2
    assert nodules[0].malignancy == 5.0
    assert len(nodules[0].centroid_patient_mm) == 3
