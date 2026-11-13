"""GIS layer loading."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path

import geopandas as gpd
import pyogrio
import shapely

from oversupply.config import Config


def read_layer(path: Path, crs: str) -> gpd.GeoDataFrame:
    """Read any OGR layer (zipped shapefiles are read in place) and reproject it.

    Date fields are skipped: some municipal exports carry corrupt dates that break parsing.
    """
    uri = f"zip://{path}" if path.suffix.lower() == ".zip" else str(path)
    info = pyogrio.read_info(uri)
    columns = [f for f, t in zip(info["fields"], info["dtypes"], strict=True)
               if not str(t).startswith("datetime")]
    return gpd.read_file(uri, columns=columns).to_crs(crs)


class Layers:
    """Lazy, cached access to the study's GIS layers, all in the projected CRS."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.crs = cfg.crs.projected

    def _load(self, name: str) -> gpd.GeoDataFrame:
        path = getattr(self.cfg.layers, name)
        if path is None:
            raise FileNotFoundError(f"Layer '{name}' is not configured (config/default.yaml)")
        return read_layer(path, self.crs)

    @cached_property
    def city_boundary(self) -> shapely.Geometry:
        return self._load("city_boundary").union_all()

    @cached_property
    def taz(self) -> gpd.GeoDataFrame:
        return self._load("taz")[["TAZ", "geometry"]]

    @cached_property
    def areas(self) -> gpd.GeoDataFrame:
        return self._load("areas")[["area", "geometry"]]

    @cached_property
    def bike_lanes(self) -> gpd.GeoDataFrame:
        return self._load("bike_lanes")

    @cached_property
    def docking_stations(self) -> gpd.GeoDataFrame:
        return self._load("docking_stations")

    @cached_property
    def neighbourhoods(self) -> gpd.GeoDataFrame:
        return self._load("neighbourhoods")

    @cached_property
    def open_space(self) -> shapely.Geometry:
        return self._load("open_space").union_all()
