# GIS layers

Source: **Tel Aviv-Yafo Municipality** (GIS department), as provided for the thesis research in
2023. Redistributed here with attribution to the municipality; all rights in the data remain
with it. These files are **not** covered by the repository's MIT licence (which applies to code).

| file | layer | CRS | used for |
|---|---|---|---|
| `city_boundary.zip` | city limits (2022) | EPSG:2039 | keep trips inside the city |
| `taz.zip` | traffic analysis zones, Tel Aviv-Yafo (221 zones) | EPSG:2039 | trip origin / destination zone |
| `areas.zip` | four city areas (North, Center, South, East) | EPSG:32636 | parking results by area |
| `bike_lanes.zip` | cycling infrastructure (Feb 2023, 811 segments) | EPSG:2039 | distance to infrastructure |
| `docking_stations.zip` | scooter docking stations (file dated June 2023, 557 polygons, no capacity field) | EPSG:2039 | separate parking inside and outside docks |

Shapefiles zipped as-is; read them in place with `geopandas.read_file("data/layers/<file>.zip")`.
