# SubTherm / OceanEmbed prototype (Team Echelon, SIH 2026, PS 26066)

Interactive Streamlit console for daily 0-1000 m subsurface temperature of the North Indian Ocean,
with per-depth uncertainty. **Demo mode: all fields are synthetic and nothing is trained.**

## Run
```
pip install -r requirements.txt      # torch is optional (needed for the H-SCAN engine and lab)
streamlit run app.py
```
Keep the `.streamlit/config.toml` next to `app.py`; it sets the light "chart room" theme.

## Files
- `app.py`       Streamlit UI: console (map, click-to-profile, A/B compare), vertical section, validation, H-SCAN lab, about.
- `emulator.py`  Synthetic ocean: surface fields, subsurface reference, demo reconstruction, D20 and heat-content metrics.
- `hscan.py`     Real PyTorch H-SCAN (CBAM + dilated ResNet + mean/log-variance heads) and the blueprint losses. Untrained.

## Using it
- Click the map to move point A (or B); the profile shows the mean line and a shaded +-2 sigma band.
- Scenarios jump to a date: Bay of Bengal barrier layer, Cyclone Amphan replay (illustrative), monsoon upwelling, marine heatwave.
- Validation tab: upload a CSV (date, lat, lon, depth_m, temp_c) of held-out INCOIS OMNI / RAMA moorings.
- H-SCAN lab: build an untrained network, run a forward pass, load a checkpoint, then switch the console engine to H-SCAN.

Checkpoints are loaded with `weights_only=True`: only tensors, lists, dicts and numbers are accepted.
