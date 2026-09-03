Fair — here it is. One line of sequencing advice, then the checklist: do this after Step 5 (drift proven) closes, not instead of it, or you'll end up demoing a pretty map in front of a monitoring system nobody's confirmed works.

## Checklist — Streamlit Frontend

### Step 1: Data prep
- [ ] Download NYC TLC `taxi_zones.geojson` (or shapefile) — same zone_id scheme as your pipeline
- [ ] Confirm your FastAPI `/predict` endpoint exists and returns demand for a given zone_id + timestamp; if not, add it first
- [ ] Decide: live per-zone API calls on load, or one batch endpoint that returns all ~260 zones at once (`/predict_all` or similar) — batch is simpler and faster for a map render, build that if it doesn't exist

### Step 2: Environment
- [ ] `pip install streamlit pydeck` (or `folium` + `streamlit-folium` if you're more familiar with folium)
- [ ] `streamlit run streamlit_app.py` — confirm hello-world renders before adding logic

### Step 3: Core map view
- [ ] Load geojson, merge with your batch prediction output on `zone_id`
- [ ] Render choropleth colored by predicted demand (pydeck `GeoJsonLayer` or plotly `choropleth_mapbox`)
- [ ] Add a color legend / scale

### Step 4: Interactivity
- [ ] Sidebar: date/hour picker → re-calls API, re-renders map
- [ ] Click or select a zone → show that zone's demand time series (line chart, last 24-48h from your prediction logs)

### Step 5: Surface the MLOps layer (this is the part that makes it a portfolio piece, not just a map)
- [ ] Small status panel: current model version, last retrain timestamp, drift status (OK/WATCH/WARNING/CRITICAL) — pull straight from your `AlertManager`/`MetricsMonitor` logs
- [ ] If severity isn't OK, show it visibly (red banner) — this is the "look, the monitoring actually does something" proof for a screen-share

### Step 6: Resilience for demo day
- [ ] Cache last-known-good predictions locally so the app doesn't break if the API is down mid-interview
- [ ] Add a loading spinner / graceful error message if the API call fails — don't let it stack-trace on screen

### Step 7: Deploy
- [ ] Streamlit Community Cloud (free, easiest) or bundle into the same Docker/Render setup as the API
- [ ] Add the live link + a GIF/screenshot to your README

That's the full build. Go.