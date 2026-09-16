# Rotation kernel tutorials

- `hotspot_wind_kernel_explained.ipynb`: Derives and validates `HotspotWindRotationKernel`
  (`starships.spectrum`) -- an emission rotation kernel for a planet with a displaced hotspot
  and a 3-component wind field (solid rotation, an equatorial jet, a day-to-night flow), built
  numerically since a 2D brightness map and a latitude-dependent wind leave no closed-form
  shortcut (unlike `CitrusRotationKernel`). No petitRADTRANS needed, runs anywhere `starships`
  is installed.
