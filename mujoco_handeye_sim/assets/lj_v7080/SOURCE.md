# LJ-V7080 mesh provenance

`LJ-V7080_3.stl` is a tessellated cache of the user-supplied
`real_laser_handeye/LJ-V7080_3.stp`. It was exported with 0.25 mm linear and
0.15 rad angular deflection. The STEP coordinates and resulting STL coordinates
are in millimetres; the MJCF mesh asset applies scale `0.001`.

The original STEP file remains the authoritative input. Rebuild with
`--force-cad` after installing the optional CAD dependencies to regenerate it.
The example YAML also pins the STEP SHA-256, so swapping only the STEP file can
never silently reuse this stale cache.
