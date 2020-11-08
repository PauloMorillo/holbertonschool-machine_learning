-- This script lists all bands with Glam rock as their main style, ranked by their longevity
SELECT split, band_name, IFNUll(split, 2020) - formed AS lifespan FROM `metal_bands` WHERE metal_bands.style LIKE '%Glam rock%' ORDER BY lifespan DESC
